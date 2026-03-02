"""
Correction strategies for PipeFusion bubble filling.

When using patch rotation to eliminate pipeline bubbles, the first patch
on non-first stages needs a predicted input (since the previous stage
hasn't computed it yet). These correction strategies provide that prediction.

The correction cache is populated during the synchronous warmup phase.
When the runtime transitions from sync to async (patch) mode, the cache
automatically splits full-sequence tensors into per-patch chunks via
``_check_patch_mode``.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import TYPE_CHECKING

from vllm_omni.diffusion.distributed.pipefusion_runtime import get_runtime_state

if TYPE_CHECKING:
    from torch import Tensor


class DirectReuse:
    # Simply reuse the last observed value.

    def __init__(self):
        # {tensor name: [{order: tensor value}]}, list position = patch id
        self.cache: defaultdict[str, list[dict[int, Tensor]]] = defaultdict(lambda: [{}])
        self.patch_mode = False

    def _check_patch_mode(self) -> int:
        runtime = get_runtime_state()
        patch_mode = runtime.patch_mode
        if self.patch_mode != patch_mode:
            if patch_mode:  # Split each cached full-sequence tensor into per-patch chunks
                num_patches = runtime.num_pipeline_patch
                for name, val in self.cache.items():
                    full_dict = val[0]  # full sequence is at patch id 0
                    # Split each tensor using runtime.split_sequence
                    # (handles both contiguous temporal and non-contiguous height splits)
                    split_lists: list[list[tuple[int, Tensor]]] = [[] for _ in range(num_patches)]
                    for order, tensor in full_dict.items():
                        splits = runtime.split_sequence(tensor, dim=-2)
                        for patch_i, split_t in enumerate(splits):
                            split_lists[patch_i].append((order, split_t))
                    self.cache[name] = [dict(pairs) for pairs in split_lists]
            else:  # Reset cache when returning to sync mode (new sample)
                self.cache = defaultdict(lambda: [{}])
            self.patch_mode = patch_mode
        return runtime.pipeline_patch_idx

    def update(self, name: str, value: Tensor, **kwargs) -> None:
        patch_id = self._check_patch_mode()
        self.cache[name][patch_id][0] = value

    def forecast(self, name: str, **kwargs) -> Tensor:
        patch_id = self._check_patch_mode()
        return self.cache[name][patch_id][0]


class TaylorSeer(DirectReuse):
    # Taylor-series finite-difference extrapolation for correction.

    def __init__(self, max_order: int = 3):
        super().__init__()
        self.order = max_order
        self.current_step, self.last_non_approximated_step = [-1], [-1]

    def _check_patch_mode(self) -> int:
        runtime = get_runtime_state()
        patch_mode = runtime.patch_mode
        if self.patch_mode != patch_mode:
            if patch_mode:  # split cache for async execution
                num_patches = runtime.num_pipeline_patch
                for name, val in self.cache.items():
                    full_dict = val[0]  # full sequence is at patch id 0
                    # Split each tensor using runtime.split_sequence
                    # (handles both contiguous temporal and non-contiguous height splits)
                    split_lists: list[list[tuple[int, Tensor]]] = [[] for _ in range(num_patches)]
                    for order, tensor in full_dict.items():
                        splits = runtime.split_sequence(tensor, dim=-2)
                        for patch_i, split_t in enumerate(splits):
                            split_lists[patch_i].append((order, split_t))
                    self.cache[name] = [dict(pairs) for pairs in split_lists]
                self.current_step = self.current_step * num_patches
                self.last_non_approximated_step = self.last_non_approximated_step * num_patches
            else:
                self.cache = defaultdict(lambda: [{}])
                self.current_step, self.last_non_approximated_step = [-1], [-1]
            self.patch_mode = patch_mode
        return runtime.pipeline_patch_idx

    def update(self, name: str, value: Tensor, **kwargs) -> None:
        patch_id = self._check_patch_mode()

        self.current_step[patch_id] += 1
        distance = self.current_step[patch_id] - self.last_non_approximated_step[patch_id]

        updated_cache = {0: value}
        cache = self.cache[name][patch_id]
        for i in range(self.order):
            if cache.get(i, None) is not None:
                updated_cache[i + 1] = (updated_cache[i] - cache[i]) / distance

        self.cache[name][patch_id] = updated_cache
        self.last_non_approximated_step[patch_id] = self.current_step[patch_id]

    def forecast(self, name: str, **kwargs) -> Tensor:
        patch_id = self._check_patch_mode()

        self.current_step[patch_id] += 1
        distance = self.current_step[patch_id] - self.last_non_approximated_step[patch_id]

        output = 0
        cache = self.cache[name][patch_id]
        for i in range(len(cache)):
            output += (1 / math.factorial(i)) * cache[i] * (distance**i)
        return output
