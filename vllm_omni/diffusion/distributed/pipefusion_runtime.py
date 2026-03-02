from __future__ import annotations

from typing import Literal

import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.distributed.parallel_state import get_pipeline_parallel_world_size, get_pp_group

logger = init_logger(__name__)

_RUNTIME: DiTRuntimeState | None = None


class DiTRuntimeState:
    patch_mode: bool
    pipeline_patch_idx: int
    pp_patches_height: list[int] | None
    pp_patches_start_end_idx: list[tuple[int, int]] | None
    pp_patches_token_num: list[int] | None
    pp_patches_token_start_end_idx: list[tuple[int, int]] | None

    def __init__(
        self,
        patch_size: tuple[int, int, int] = (1, 2, 2),
        warmup_steps: int = 1,
        split_dim: Literal["height", "temporal"] = "height",
        use_bubble_filling: bool = False,
    ):
        self.patch_size = patch_size
        self.patch_mode = False
        self.pipeline_patch_idx = 0
        self.warmup_steps = warmup_steps
        self.split_dim = split_dim  # "height" or "temporal"
        self.use_bubble_filling = use_bubble_filling

    def set_input_parameters(self, latents: torch.Tensor, dtype):
        self._calc_patches_metadata(latents)
        self._reset_recv_buffer(dtype)

    def set_patched_mode(self, patch_mode: bool):
        self.patch_mode = patch_mode
        self.pipeline_patch_idx = 0

    def next_patch(self, patch_idx: int | None = None):
        if patch_idx is not None:
            self.pipeline_patch_idx = patch_idx
        elif self.patch_mode:
            self.pipeline_patch_idx += 1
            if self.pipeline_patch_idx == self.num_pipeline_patch:
                self.pipeline_patch_idx = 0
        else:
            self.pipeline_patch_idx = 0

    def _calc_patch_metadata(self, seq_length):
        base_len, remainder = seq_length // self.num_pipeline_patch, seq_length % self.num_pipeline_patch
        lengths = [base_len + int(i < remainder) for i in range(self.num_pipeline_patch)]
        start = 0
        start_end_idx = []
        for num in lengths:
            start_end_idx.append((start, start + num))
            start += num
        return lengths, start_end_idx

    def _calc_patches_metadata(self, latents):
        self.num_pipeline_patch = get_pipeline_parallel_world_size()

        p_t, p_h, p_w = self.patch_size
        ppf = latents.size(-3) // p_t  # post-patch frames
        pph = latents.size(-2) // p_h  # post-patch height (full)
        ppw = latents.size(-1) // p_w  # post-patch width

        # Store post-patch spatial dims for KV cache reshape in attention
        self.ppf = ppf
        self.pph = pph
        self.ppw = ppw

        if self.split_dim == "height":
            # Split along spatial height
            self.latent_split_dim = -2  # dim in 5D [B,C,T,H,W]

            # Post-patch heights split among patches
            self.pp_patches_post_height, self.pp_patches_post_start_end_idx = self._calc_patch_metadata(pph)
            self.pp_patches_post_frames = None  # not split

            # Latent-space heights (multiply by p_h to ensure divisibility)
            self.pp_patches_height = [h * p_h for h in self.pp_patches_post_height]
            start = 0
            self.pp_patches_start_end_idx = []
            for h in self.pp_patches_height:
                self.pp_patches_start_end_idx.append((start, start + h))
                start += h

            # Token count: each patch covers all frames and widths but partial height
            self.pp_patches_token_num = [h * ppw * ppf for h in self.pp_patches_post_height]

        elif self.split_dim == "temporal":
            # Split along temporal (frames) dimension
            self.latent_split_dim = -3  # dim in 5D [B,C,T,H,W]

            # Post-patch frames split among patches
            self.pp_patches_post_frames, self.pp_patches_post_start_end_idx = self._calc_patch_metadata(ppf)
            self.pp_patches_post_height = None  # not split

            # Latent-space frames (multiply by p_t to ensure divisibility)
            self.pp_patches_height = [f * p_t for f in self.pp_patches_post_frames]
            start = 0
            self.pp_patches_start_end_idx = []
            for f in self.pp_patches_height:
                self.pp_patches_start_end_idx.append((start, start + f))
                start += f

            # Token count: each patch covers all heights and widths but partial frames
            self.pp_patches_token_num = [f * pph * ppw for f in self.pp_patches_post_frames]

        else:
            raise ValueError(f"Unknown split_dim: {self.split_dim}. Use 'height' or 'temporal'.")

        # Calculate start/end indices for each patch's tokens
        start = 0
        self.pp_patches_token_start_end_idx = []
        for num in self.pp_patches_token_num:
            self.pp_patches_token_start_end_idx.append((start, start + num))
            start += num

    def split_sequence(self, tensor: torch.Tensor, dim: int = -2) -> list[torch.Tensor]:
        """Split a token-sequence tensor into per-patch chunks along the correct dimension.

        For temporal splitting, tokens are contiguous in [f, h, w] order so a
        simple ``tensor.split(token_nums, dim)`` works.

        For height splitting, each patch's tokens are interleaved across frames
        and NOT contiguous. We reshape to [B, ppf, pph, ppw, ...], slice along
        the height axis, then flatten back.

        Args:
            tensor: Token-space tensor, e.g. [B, seq, D] with seq = ppf*pph*ppw.
            dim: The sequence dimension in *tensor* (default -2).
        """
        if self.split_dim == "temporal":
            return list(tensor.split(self.pp_patches_token_num, dim=dim))

        # Height split — need 5D view
        # tensor shape: [B, ppf*pph*ppw, *rest]
        rest = tensor.shape[dim + 1 :] if dim != -1 else ()  # dims after seq
        if dim < 0:
            dim = tensor.ndim + dim

        # Reshape seq → (ppf, pph, ppw)
        new_shape = list(tensor.shape[:dim]) + [self.ppf, self.pph, self.ppw] + list(rest)
        tensor_5d = tensor.view(new_shape)

        # Split along pph (which is at position dim+1 in the reshaped tensor)
        height_dim = dim + 1
        splits = tensor_5d.split(self.pp_patches_post_height, dim=height_dim)

        # Flatten (ppf, pph_patch, ppw) back to seq for each split
        result = []
        for s in splits:
            flat_shape = list(s.shape[:dim]) + [-1] + list(rest)
            result.append(s.reshape(flat_shape))
        return result

    def _reset_recv_buffer(self, dtype):
        get_pp_group().reset_buffer()
        get_pp_group().set_config(dtype)


def initialize_runtime_state(
    patch_size: tuple[int, int, int] = (1, 2, 2),
    warmup_steps: int = 1,
    split_dim: Literal["height", "temporal"] = "height",
    use_bubble_filling: bool = False,
):
    global _RUNTIME
    if _RUNTIME is not None:
        logger.warning("Runtime state is already initialized, reinitializing with pipeline...")
    _RUNTIME = DiTRuntimeState(
        patch_size=patch_size,
        warmup_steps=warmup_steps,
        split_dim=split_dim,
        use_bubble_filling=use_bubble_filling,
    )


def get_runtime_state():
    assert _RUNTIME is not None, "Runtime state has not been initialized."
    return _RUNTIME
