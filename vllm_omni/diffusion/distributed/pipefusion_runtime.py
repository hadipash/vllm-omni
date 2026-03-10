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
    ):
        self.patch_size = patch_size
        self.patch_mode = False
        self.pipeline_patch_idx = 0
        self.warmup_steps = warmup_steps
        self.split_dim = split_dim  # "height" or "temporal"

    def set_input_parameters(self, latents: torch.Tensor, dtype):
        self._calc_patches_metadata(latents)
        self._reset_recv_buffer(dtype)

    def set_patched_mode(self, patch_mode: bool):
        self.patch_mode = patch_mode
        self.pipeline_patch_idx = 0

    def next_patch(self):
        if self.patch_mode:
            self.pipeline_patch_idx += 1
            if self.pipeline_patch_idx == self.num_pipeline_patch:
                self.pipeline_patch_idx = 0
        else:
            self.pipeline_patch_idx = 0

    def _calc_patch_metadata(self, seq_length):
        lengths = [seq_length // self.num_pipeline_patch] * (self.num_pipeline_patch - 1)
        # Give more tokens to the last patch, if it's the case.
        lengths.append(seq_length // self.num_pipeline_patch + seq_length % self.num_pipeline_patch)
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

    def _reset_recv_buffer(self, dtype):
        get_pp_group().reset_buffer()
        get_pp_group().set_config(dtype)


def initialize_runtime_state(
    patch_size: tuple[int, int, int] = (1, 2, 2),
    warmup_steps: int = 1,
    split_dim: Literal["height", "temporal"] = "height",
):
    global _RUNTIME
    if _RUNTIME is not None:
        logger.warning("Runtime state is already initialized, reinitializing with pipeline...")
    _RUNTIME = DiTRuntimeState(patch_size=patch_size, warmup_steps=warmup_steps, split_dim=split_dim)


def get_runtime_state():
    assert _RUNTIME is not None, "Runtime state has not been initialized."
    return _RUNTIME
