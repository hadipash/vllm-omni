from __future__ import annotations

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

    def __init__(self, warmup_steps: int = 1):
        self.patch_mode = False
        self.pipeline_patch_idx = 0
        self.warmup_steps = warmup_steps

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
        self.pp_patches_height, self.pp_patches_start_end_idx = self._calc_patch_metadata(latents.size(-2))
        # FIXME: patch size
        seq_length = latents.size(-1) // 2 * latents.size(-2) // 2 * latents.size(-3)
        self.pp_patches_token_num, self.pp_patches_token_start_end_idx = self._calc_patch_metadata(seq_length)

    def _reset_recv_buffer(self, dtype):
        get_pp_group().reset_buffer()
        get_pp_group().set_config(dtype)


def initialize_runtime_state():
    global _RUNTIME
    if _RUNTIME is not None:
        logger.warning("Runtime state is already initialized, reinitializing with pipeline...")
    _RUNTIME = DiTRuntimeState()


def get_runtime_state():
    assert _RUNTIME is not None, "Runtime state has not been initialized."
    return _RUNTIME
