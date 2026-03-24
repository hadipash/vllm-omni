# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
PipeFusion scheduler mixin for diffusion models.

Provides patch-wise cache management for schedulers in PipeFusion mode:
- Splitting scheduler state (model outputs, cached samples) into per-patch versions
- Swapping per-patch state in/out during each scheduler step
- Gating step-level state advancement to only the last patch

Usage:
    class MyScheduler(SchedulerMixin, ConfigMixin, PipeFusionSchedulerMixin, BaseScheduler):
        # Declare which attributes need per-patch caching:
        #   "list" = list[Tensor | None], each element split independently
        #   "tensor" = single Tensor | None, split directly
        _pipefusion_patch_cache_spec: ClassVar[list[tuple[str, str]]] = [
            ("model_outputs", "list"),
            ("last_sample", "tensor"),
        ]

        def __init__(self, ...):
            ...
            self.pipefusion_init_patch_caches()

        def step(self, model_output, timestep, sample, ...):
            ...
            patch_mode, patch_idx, is_last_patch = self.pipefusion_get_patch_context()
            if patch_mode:
                self.pipefusion_load_patch_state(patch_idx)

            ... (main solver logic) ...

            # For tensor caches that get reassigned (not mutated in-place):
            self.pipefusion_update_value("last_sample", patch_idx, sample)
            self.last_sample = sample

            ... (compute prev_sample) ...

            # Only advance step-level state on the last patch
            if is_last_patch:
                self._step_index += 1
"""

from __future__ import annotations

from typing import Any, ClassVar

import torch

from vllm_omni.diffusion.distributed.pipefusion_runtime import get_runtime_state


class PipeFusionSchedulerMixin:
    """
    Mixin for schedulers that participate in PipeFusion patch-wise execution.

    In async PipeFusion mode, the scheduler processes multiple spatial patches
    per timestep. Each patch needs its own copy of certain scheduler state
    (e.g., cached model outputs, previous samples) while sharing others
    (e.g., step index, timestep list).

    Subclasses declare per-patch cached attributes via `_pipefusion_patch_cache_spec`
    and use the provided helpers to manage state during `step()`.
    """

    # Override in subclass: list of (attr_name, cache_type)
    #   cache_type "list" = list[Tensor | None], each element split independently
    #   cache_type "tensor" = single Tensor | None, split directly
    _pipefusion_patch_cache_spec: ClassVar[list[tuple[str, str]]] = []

    def pipefusion_init_patch_caches(self) -> None:
        """Initialize per-patch cache storage. Call during __init__."""
        self._pf_patch_caches: dict[str, list[Any]] | None = None

    def split_caches_for_patches(self, patch_sizes: list[int], dim: int = -2) -> None:
        """
        Split cached scheduler state into per-patch versions for async pipeline.

        Called when transitioning from sync to async pipeline mode.
        Each attribute listed in `_pipefusion_patch_cache_spec` is split
        along the specified dimension.

        Args:
            patch_sizes: Size of each patch along the split dimension.
            dim: Dimension along which to split (default -2 for height).
        """
        num_patches = get_runtime_state().num_pipeline_patch
        self._pf_patch_caches = {}

        for attr_name, cache_type in self._pipefusion_patch_cache_spec:
            value = getattr(self, attr_name)

            if cache_type == "list":
                # list[Tensor | None] — split each non-None element
                per_patch: list[list[torch.Tensor | None]] = []
                for patch_idx in range(num_patches):
                    patch_list: list[torch.Tensor | None] = []
                    for item in value:
                        if item is not None:
                            splits = item.split(patch_sizes, dim=dim)
                            patch_list.append(splits[patch_idx])
                        else:
                            patch_list.append(None)
                    per_patch.append(patch_list)
                self._pf_patch_caches[attr_name] = per_patch

            elif cache_type == "tensor":
                # single Tensor | None — split directly
                if value is not None:
                    splits = value.split(patch_sizes, dim=dim)
                    self._pf_patch_caches[attr_name] = list(splits)
                else:
                    self._pf_patch_caches[attr_name] = [None] * num_patches

    def clear_patch_caches(self) -> None:
        """Clear per-patch caches when exiting async pipeline mode."""
        self._pf_patch_caches = None

    def merge_caches_from_patches(self, dim: int = -2) -> None:
        if self._pf_patch_caches is None:
            return

        for attr_name, cache_type in self._pipefusion_patch_cache_spec:
            if attr_name not in self._pf_patch_caches:
                continue

            value = self._pf_patch_caches[attr_name]

            if cache_type == "list":
                merged_value = []
                num_items = len(value[0]) if value else 0
                for item_idx in range(num_items):
                    patch_values = [patch_value[item_idx] for patch_value in value]
                    if patch_values[0] is None:
                        merged_value.append(None)
                    else:
                        merged_value.append(torch.cat(patch_values, dim=dim))
                setattr(self, attr_name, merged_value)
            elif cache_type == "tensor":
                if not value or value[0] is None:
                    setattr(self, attr_name, None)
                else:
                    setattr(self, attr_name, torch.cat(value, dim=dim))

        self._pf_patch_caches = None

    def pipefusion_step_begin(self) -> tuple[int, bool]:
        """
        Begin a scheduler step: get patch context and load per-patch state.

        Combines patch context lookup and state loading into a single call.
        In patch mode, swaps cached attributes to the current patch's versions.

        Returns:
            (patch_idx, is_last_patch) tuple:
            - patch_idx: Current patch index (0 if not in patch mode).
            - is_last_patch: Whether this is the last patch in the sequence.
        """
        runtime_state = get_runtime_state()
        patch_mode = runtime_state.patch_mode
        patch_idx = runtime_state.pipeline_patch_idx if patch_mode else 0
        is_last_patch = not patch_mode or patch_idx == runtime_state.num_pipeline_patch - 1

        if patch_mode and self._pf_patch_caches is not None:
            for attr_name, _ in self._pipefusion_patch_cache_spec:
                if attr_name in self._pf_patch_caches:
                    setattr(self, attr_name, self._pf_patch_caches[attr_name][patch_idx])

        return patch_idx, is_last_patch

    def pipefusion_update_value(self, attr_name: str, value: Any) -> None:
        """
        Save a value back to the per-patch cache.

        Needed for "tensor" caches where assignment rebinds the attribute
        rather than mutating it in-place. "list" caches that are mutated
        in-place (e.g., shifting elements) are updated automatically since
        `pipefusion_step_begin` sets self.attr to the actual list object.

        Args:
            attr_name: Name of the cached attribute.
            patch_idx: Index of the patch to save for.
            value: The value to save.
        """
        runtime_state = get_runtime_state()
        patch_mode = runtime_state.patch_mode
        patch_idx = runtime_state.pipeline_patch_idx if patch_mode else 0

        if self._pf_patch_caches is not None and attr_name in self._pf_patch_caches:
            self._pf_patch_caches[attr_name][patch_idx] = value
