# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
PipeFusion transformer mixin for diffusion models.

Provides patch-wise execution abstractions for transformer models, handling:
- Conditional patch embedding (first stage only)
- Conditional output projection & unpatchify (last stage only)
- RoPE slicing for patch-wise execution
- KV cache management for self-attention across patches
- Conv3d activation caching for patch boundaries
- Pipeline-parallel weight loading with layer remapping
"""

import re

import torch
import torch.nn.functional as F

from vllm_omni.diffusion.distributed.parallel_state import (
    get_pipeline_parallel_rank,
    get_pipeline_parallel_world_size,
    is_pipeline_first_stage,
    is_pipeline_last_stage,
)
from vllm_omni.diffusion.distributed.pipefusion_runtime import get_runtime_state


class PipeFusionTransformerMixin:
    """
    Mixin for transformer models that participate in PipeFusion.

    Provides helper methods for:
    1. Slicing RoPE embeddings to match the current patch
    2. Conditional patch embedding (first stage) and output projection (last stage)
    3. Pipeline-parallel weight loading with block index remapping

    Models using this mixin should call these helpers from their forward() method
    instead of inlining the PipeFusion logic.
    """

    def pipefusion_slice_rotary_emb(
        self,
        rotary_emb: tuple[torch.Tensor, ...],
        dims: tuple[int, ...],
        patch_size: tuple[int, int, int],
    ) -> tuple[torch.Tensor, ...]:
        """
        Slice RoPE embeddings for the current pipeline patch.

        In patch mode, RoPE is computed for the full spatial extent, then
        sliced along the split dimension to match the current patch.

        Args:
            rotary_emb: Full RoPE tuple (freqs_cos, freqs_sin).
            dims: Original latent dimensions (B, C, T, H, W).
            patch_size: Model's 3D patch size (p_t, p_h, p_w).

        Returns:
            Sliced RoPE tuple for the current patch.
        """
        runtime = get_runtime_state()
        if not runtime.patch_mode:
            return rotary_emb

        p_t, p_h, p_w = patch_size
        ppf = dims[2] // p_t  # post-patch frames
        pph = dims[3] // p_h  # post-patch height (full)
        ppw = dims[4] // p_w  # post-patch width

        patch_idx = runtime.pipeline_patch_idx

        if runtime.split_dim == "temporal":
            # Split along frames (dim=0 in [ppf, pph, ppw])
            split_dim = 0
            pp_sizes = runtime.pp_patches_post_frames
        else:
            # Split along height (dim=1 in [ppf, pph, ppw])
            split_dim = 1
            pp_sizes = runtime.pp_patches_post_height

        def split_rope(re):
            # [1, ppf*pph*ppw, 1, dim] -> [ppf, pph, ppw, dim]
            re = re.reshape(ppf, pph, ppw, -1)
            # Split along the chosen dimension and select current patch
            re = re.split(pp_sizes, dim=split_dim)[patch_idx]
            # Reshape back to [1, seq, 1, dim]
            return re.reshape(1, -1, 1, re.shape[-1])

        return tuple(split_rope(re) for re in rotary_emb)

    @staticmethod
    def pipefusion_get_post_patch_height(height: int, patch_height: int) -> int:
        """
        Get the post-patch height for the current pipeline stage.

        In height-split mode, returns the height of the current patch;
        in temporal-split mode or non-patch mode, returns the full post-patch height.
        """
        runtime = get_runtime_state()
        if runtime.patch_mode and runtime.split_dim == "height":
            return runtime.pp_patches_post_height[runtime.pipeline_patch_idx]
        else:
            return height // patch_height

    @staticmethod
    def pipefusion_get_post_patch_num_frames(num_frames: int, patch_frames: int) -> int:
        """
        Get the post-patch frame count for the current pipeline stage.

        In temporal-split mode, returns the frame count of the current patch;
        in height-split mode or non-patch mode, returns the full post-patch frame count.
        """
        runtime = get_runtime_state()
        if runtime.patch_mode and runtime.split_dim == "temporal":
            return runtime.pp_patches_post_frames[runtime.pipeline_patch_idx]
        else:
            return num_frames // patch_frames

    @staticmethod
    def pipefusion_should_patch_embed() -> bool:
        """Whether this rank should run the patch embedding (first stage only)."""
        return is_pipeline_first_stage()

    @staticmethod
    def pipefusion_should_output_project() -> bool:
        """Whether this rank should run output norm/projection/unpatchify (last stage only)."""
        return is_pipeline_last_stage()

    def pipefusion_reset_caches(self) -> None:
        """
        Reset all PipeFusion caches (KV caches in attention, activation caches in Conv3d).

        Must be called at the start of each new diffusion request to prevent
        stale data from a previous run (e.g. the dummy warmup run) from
        contaminating the current run.
        """
        from vllm_omni.diffusion.distributed.correction import DirectReuse

        for module in self.modules():
            if isinstance(module, PipeFusionSelfAttentionMixin):
                module._kv_caches = {}
            if isinstance(module, PipeFusionConv3dMixin):
                module.activation_cache = None
        # Reset correction cache (bubble filling) if attached
        correction = getattr(self, "correction", None)
        if isinstance(correction, DirectReuse):
            correction.cache.clear()
            correction.patch_mode = False

    # Class-level attribute: name of the block container to split for PP.
    # Subclasses can override this if their blocks attribute is named differently.
    _pp_block_name: str = "blocks"

    def pipefusion_split_blocks(self) -> None:
        """
        Split transformer blocks across pipeline-parallel ranks.

        Assigns a contiguous subset of blocks to this rank and stores
        the offset/count for weight remapping. Later ranks get more blocks
        since the first stage usually has extra processing (patch embed).
        """
        pp_rank = get_pipeline_parallel_rank()
        pp_world_size = get_pipeline_parallel_world_size()

        block_name = self._pp_block_name
        blocks = getattr(self, block_name)
        num_blocks = len(blocks)

        num_per_stage = num_blocks // pp_world_size
        remainder = num_blocks % pp_world_size
        # Give more blocks to earlier stages
        start = pp_rank * num_per_stage + min(pp_rank, remainder)
        end = (pp_rank + 1) * num_per_stage + min(pp_rank + 1, remainder)

        setattr(self, block_name, blocks[start:end])
        self.pp_layer_offset = start
        self.pp_num_layers = end - start

    def pipefusion_remap_block_weights(
        self,
        name: str,
        block_prefix: str = "blocks",
    ) -> str | None:
        """
        Remap checkpoint block index to local block index for pipeline parallel.

        Args:
            name: Weight parameter name (e.g., "blocks.15.attn1.to_qkv.weight").
            block_prefix: Prefix for transformer blocks (default "blocks").

        Returns:
            Remapped name with local block index, or None if this weight
            doesn't belong to this rank.
        """
        pp_layer_offset = getattr(self, "pp_layer_offset", 0)
        pp_num_layers = getattr(self, "pp_num_layers", None)

        if pp_num_layers is None:
            return name

        pattern = rf"{block_prefix}\.(\d+)\.(.*)"
        match = re.match(pattern, name)
        if not match:
            return name

        global_layer_idx = int(match.group(1))
        local_layer_idx = global_layer_idx - pp_layer_offset

        # Skip if this layer doesn't belong to this rank
        if local_layer_idx < 0 or local_layer_idx >= pp_num_layers:
            return None

        # Remap to local index
        return f"{block_prefix}.{local_layer_idx}.{match.group(2)}"


class PipeFusionSelfAttentionMixin:
    """
    Mixin for self-attention modules that participate in PipeFusion.

    In patch mode, maintains full K/V caches across patches so that
    each patch's query can attend to the full sequence.

    Maintains separate KV caches for conditional ("inputs") and
    unconditional ("inputs_uncond") predictions to prevent CFG
    negative predictions from contaminating the conditional cache.
    The active cache is selected by the parent transformer's
    ``cache_key`` attribute.
    """

    # Cache dict: cache_key -> (full_k, full_v)
    _kv_caches: dict[str, tuple[torch.Tensor, torch.Tensor]]

    def _get_kv_cache(self, cache_key: str) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        return self._kv_caches[cache_key]

    def _set_kv_cache(self, cache_key: str, k: torch.Tensor, v: torch.Tensor) -> None:
        """Store the KV cache for the given correction key."""
        if not hasattr(self, "_kv_caches"):
            self._kv_caches = {}
        self._kv_caches[cache_key] = (k, v)

    def pipefusion_update_kv_cache(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Update and return full K/V for the current patch.

        In patch mode, inserts the current patch's K/V into the full cache
        and returns the full K/V. In non-patch mode, just stores K/V directly.

        Uses the parent transformer's ``cache_key`` to select between
        separate conditional / unconditional KV caches, preventing the CFG
        negative prediction from overwriting the conditional cache.

        The token sequence is flattened as [frames, height, width], so
        height-based patches are NOT contiguous in the flat sequence.
        We reshape to 5D [B, ppf, pph, ppw, heads, dim] to write
        at the correct height positions via view-based slicing.

        Args:
            key: Current patch's key tensor [B, patch_seq, heads, dim].
            value: Current patch's value tensor [B, patch_seq, heads, dim].

        Returns:
            (full_key, full_value) for attention computation.
        """
        runtime = get_runtime_state()
        cache_key = getattr(self, "_parent_cache_key", "inputs")

        if runtime.patch_mode:
            full_k, full_v = self._get_kv_cache(cache_key)
            ppf, pph, ppw = runtime.ppf, runtime.pph, runtime.ppw
            patch_start, patch_end = runtime.pp_patches_post_start_end_idx[runtime.pipeline_patch_idx]
            B, _, heads, dim = key.shape

            if runtime.split_dim == "temporal":
                # Temporal split: tokens are contiguous in [f, h, w] order
                # because frames are the outermost dimension.
                # Patch covers f∈[f_start, f_end), token range is contiguous.
                tok_start = patch_start * pph * ppw
                tok_end = patch_end * pph * ppw
                full_k[:, tok_start:tok_end] = key
                full_v[:, tok_start:tok_end] = value
            else:
                # Height split: tokens are NON-contiguous (interleaved by frames).
                # Reshape to 5D [B, ppf, pph, ppw, heads, dim] and slice height.
                pph_patch = patch_end - patch_start
                key_5d = key.view(B, ppf, pph_patch, ppw, heads, dim)
                value_5d = value.view(B, ppf, pph_patch, ppw, heads, dim)
                full_k_5d = full_k.view(B, ppf, pph, ppw, heads, dim)
                full_v_5d = full_v.view(B, ppf, pph, ppw, heads, dim)
                full_k_5d[:, :, patch_start:patch_end, :, :, :] = key_5d
                full_v_5d[:, :, patch_start:patch_end, :, :, :] = value_5d

            return full_k, full_v
        else:
            self._set_kv_cache(cache_key, key, value)
            return key, value


class PipeFusionConv3dMixin:
    """
    Mixin for Conv3d layers that participate in PipeFusion.

    In patch mode, maintains an activation cache so that convolution
    at patch boundaries can access neighboring patches' activations.
    """

    def pipefusion_conv3d_enabled(self) -> bool:
        """Whether this conv layer should use PipeFusion patch-wise execution.

        Only needed when kernel != stride (overlapping convolutions that
        require boundary data from neighbouring patches).  When kernel == stride
        (e.g. the patch embedding), each output position depends on exactly one
        non-overlapping input block, so the direct conv on the patch is correct.
        """
        runtime = get_runtime_state()
        return runtime.patch_mode and runtime.num_pipeline_patch > 1 and self.kernel_size != self.stride

    def pipefusion_conv3d_forward(self, x: torch.Tensor, dims: tuple[int, ...]) -> torch.Tensor:
        """
        Forward pass for Conv3d with PipeFusion patch support.

        In patch mode, caches activations and performs sliced convolution
        to handle boundary conditions correctly. Call only when
        `pipefusion_conv3d_enabled()` returns True.

        Args:
            x: Input tensor for the current patch.
            dims: Original full input dimensions.

        Returns:
            Convolution output for the current patch.
        """
        runtime = get_runtime_state()

        if getattr(self, "activation_cache", None) is None:
            self.activation_cache = torch.zeros(dims, dtype=x.dtype, device=x.device)

        patch_idx = runtime.pipeline_patch_idx
        start, end = runtime.pp_patches_start_end_idx[patch_idx]
        out_start, out_end = runtime.pp_patches_post_start_end_idx[patch_idx]
        self.activation_cache[:, :, :, start:end, :] = x
        return self._sliced_conv3d_forward(self.activation_cache, out_start, out_end)

    def _sliced_conv3d_forward(self, x: torch.Tensor, out_start: int, out_end: int) -> torch.Tensor:
        """
        Compute convolution on a slice of the input that produces output for [out_start:out_end].

        Args:
            x: Full input tensor with all patches cached.
            out_start, out_end: Output slice range (post-patch space).
        """
        b, c, t, h, w = x.shape
        pad_t, pad_h, pad_w = self.padding
        stride_h = self.stride[1] if isinstance(self.stride, tuple) else self.stride

        # Calculate input range needed to produce output [out_start:out_end]
        # For strided conv: out_pos = (in_pos + pad - kernel_size) // stride + 1
        # Inverse: in_pos = out_pos * stride - pad (approximately)
        in_start = out_start * stride_h
        in_end = (out_end - 1) * stride_h + self.kernel_size[1]  # Need full kernel for last output

        # Expand to include padding context from neighbors
        h_begin = max(0, in_start - pad_h)
        h_end = min(h, in_end + pad_h)

        # Determine padding needed at boundaries
        pad_top = max(0, pad_h - in_start) if h_begin == 0 else 0
        pad_bottom = max(0, in_end + pad_h - h) if h_end == h else 0

        sliced_input = x[:, :, :, h_begin:h_end, :]
        padded_input = F.pad(sliced_input, (pad_w, pad_w, pad_top, pad_bottom, pad_t, pad_t), mode="constant")

        output = F.conv3d(
            padded_input,
            self.weight,
            self.bias,
            stride=self.stride,
            padding="valid",
            dilation=self.dilation,
            groups=self.groups,
        )

        # Extract only the output rows we need (in case we computed extra)
        expected_out_height = out_end - out_start
        if output.shape[3] > expected_out_height:
            output = output[:, :, :, :expected_out_height, :]

        return output
