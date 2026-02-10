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
        sliced along the height dimension to match the current patch.

        Args:
            rotary_emb: Full RoPE tuple (freqs_cos, freqs_sin).
            dims: Original latent dimensions (B, C, T, H, W).
            patch_size: Model's 3D patch size (p_t, p_h, p_w).

        Returns:
            Sliced RoPE tuple for the current patch.
        """
        if not get_runtime_state().patch_mode:
            return rotary_emb

        p_t, p_h, p_w = patch_size
        ppf = dims[2] // p_t  # post-patch frames
        pph = dims[3] // p_h  # post-patch height (full)
        ppw = dims[4] // p_w  # post-patch width

        pp_heights = get_runtime_state().pp_patches_post_height
        patch_idx = get_runtime_state().pipeline_patch_idx

        def split_rope(re):
            # [1, ppf*pph*ppw, 1, dim] -> [ppf, pph, ppw, dim]
            re = re.reshape(ppf, pph, ppw, -1)
            # Split along height (dim=1) and select current patch
            re = re.split(pp_heights, dim=1)[patch_idx]
            # Reshape back to [1, seq, 1, dim]
            return re.reshape(1, -1, 1, re.shape[-1])

        return tuple(split_rope(re) for re in rotary_emb)

    @staticmethod
    def pipefusion_get_post_patch_height(height: int, patch_height: int) -> int:
        """
        Get the post-patch height for the current pipeline stage.

        In patch mode, returns the height of the current patch; otherwise
        returns the full post-patch height.

        Args:
            height: Full spatial height.
            patch_height: Patch size along height dimension.

        Returns:
            Post-patch height for the current patch or full height.
        """
        if get_runtime_state().patch_mode:
            return get_runtime_state().pp_patches_post_height[get_runtime_state().pipeline_patch_idx]
        else:
            return height // patch_height

    @staticmethod
    def pipefusion_should_patch_embed() -> bool:
        """Whether this rank should run the patch embedding (first stage only)."""
        return is_pipeline_first_stage()

    @staticmethod
    def pipefusion_should_output_project() -> bool:
        """Whether this rank should run output norm/projection/unpatchify (last stage only)."""
        return is_pipeline_last_stage()

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
        # Give more blocks to later stages (first stage has patch embed overhead)
        start = pp_rank * num_per_stage + max(0, pp_rank - (pp_world_size - remainder))
        end = (pp_rank + 1) * num_per_stage + max(0, (pp_rank + 1) - (pp_world_size - remainder))

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
    """

    def pipefusion_update_kv_cache(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Update and return full K/V for the current patch.

        In patch mode, inserts the current patch's K/V into the full cache
        and returns the full K/V. In non-patch mode, just stores K/V directly.

        Args:
            key: Current patch's key tensor.
            value: Current patch's value tensor.

        Returns:
            (full_key, full_value) for attention computation.
        """
        if get_runtime_state().patch_mode:
            start, end = get_runtime_state().pp_patches_token_start_end_idx[get_runtime_state().pipeline_patch_idx]
            self.full_k[:, start:end] = key
            self.full_v[:, start:end] = value
            return self.full_k, self.full_v
        else:
            self.full_k = key
            self.full_v = value
            return key, value


class PipeFusionConv3dMixin:
    """
    Mixin for Conv3d layers that participate in PipeFusion.

    In patch mode, maintains an activation cache so that convolution
    at patch boundaries can access neighboring patches' activations.
    """

    def pipefusion_conv3d_enabled(self) -> bool:
        """Whether this conv layer should use PipeFusion patch-wise execution."""
        runtime = get_runtime_state()
        return (
            runtime.patch_mode
            and runtime.num_pipeline_patch > 1
            and self.kernel_size != (1, 1)
            and self.kernel_size != 1
        )

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
