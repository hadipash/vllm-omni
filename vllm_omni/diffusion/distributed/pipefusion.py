# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
PipeFusion mixin for diffusion pipelines.

Provides patch-wise execution and pipeline parallelism abstractions,
allowing models to split the denoising loop across multiple ranks with
asynchronous communication. Analogous to CFGParallelMixin for CFG parallelism.

Usage:
    class MyPipeline(nn.Module, CFGParallelMixin, PipeFusionPipelineMixin):
        def diffuse(self, ...):
            # Use self._pipefusion_sync_pipeline / self._pipefusion_async_pipeline
            ...
"""

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.distributed as dist

from vllm_omni.diffusion.distributed.parallel_state import (
    get_dit_group,
    get_pipeline_parallel_world_size,
    get_pp_group,
    is_pipeline_first_stage,
    is_pipeline_last_stage,
)
from vllm_omni.diffusion.distributed.pipefusion_runtime import get_runtime_state


class PipeFusionPipelineMixin(ABC):
    """
    Mixin class providing PipeFusion (patch-wise + pipeline parallel) logic
    for diffusion pipelines.

    Pipelines inherit this mixin and call its sync/async pipeline methods
    from their diffuse() implementation. Model-specific noise prediction
    kwargs are supplied via the `prepare_pipefusion_noise_kwargs` hook.

    Required attributes on the pipeline:
        - scheduler: A scheduler with `step()`, `split_caches_for_patches()`,
          and `clear_patch_caches()` methods.
        - _current_timestep: Tracked by the pipeline.

    Required methods (to be implemented by subclasses):
        - predict_noise_maybe_with_cfg(): From CFGParallelMixin.
        - scheduler_step_maybe_with_cfg(): From CFGParallelMixin.
        - prepare_pipefusion_noise_kwargs(): Returns (positive_kwargs, negative_kwargs)
          for the current timestep/patch.
    """

    def combine_cfg_noise(
        self, noise_pred: torch.Tensor, neg_noise_pred: torch.Tensor, true_cfg_scale: float, cfg_normalize: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Override CFGParallelMixin's combine_cfg_noise for PipeFusion.

        On the last pipeline stage, combine as usual.
        On intermediate stages, return both predictions as a tuple
        so they can be forwarded to the next stage.
        """
        if is_pipeline_last_stage():
            return neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)
        else:
            return noise_pred, neg_noise_pred

    @abstractmethod
    def prepare_pipefusion_noise_kwargs(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        do_true_cfg: bool,
        noise_uncond: torch.Tensor | None,
        **extra_kwargs: Any,
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        """
        Prepare positive and negative kwargs for noise prediction.

        Subclasses MUST implement this to provide model-specific kwargs.

        Args:
            latents: The input latents for this step/patch.
            timestep: Current timestep tensor.
            do_true_cfg: Whether CFG is active.
            noise_uncond: Unconditional noise from previous rank (for non-first stages).
            **extra_kwargs: Additional pipeline-specific arguments.

        Returns:
            (positive_kwargs, negative_kwargs) tuple for predict_noise_maybe_with_cfg.
        """
        raise NotImplementedError("Subclasses must implement prepare_pipefusion_noise_kwargs")

    def _init_sync_pipeline(self, latents: torch.Tensor) -> torch.Tensor:
        """Initialize sync pipeline mode."""
        get_runtime_state().set_patched_mode(patch_mode=False)
        return latents

    def _init_async_pipeline(self, num_timesteps: int, latents: torch.Tensor) -> list[torch.Tensor | None]:
        """Initialize async pipeline mode: split latents into patches and set up recv tasks."""
        get_runtime_state().set_patched_mode(patch_mode=True)

        split_sizes = get_runtime_state().pp_patches_height
        split_dim = get_runtime_state().latent_split_dim

        if is_pipeline_first_stage():
            # get latents computed in warmup stage
            # ignore latents after the last timestep
            latents = get_pp_group().pipeline_recv() if get_runtime_state().warmup_steps > 0 else latents
            patch_latents = list(latents.split(split_sizes, dim=split_dim))
        elif is_pipeline_last_stage():
            patch_latents = list(latents.split(split_sizes, dim=split_dim))
            # Split scheduler caches into per-patch versions for async pipeline
            self.scheduler.split_caches_for_patches(split_sizes, dim=split_dim)
        else:
            patch_latents = [None] * get_runtime_state().num_pipeline_patch

        recv_timesteps = num_timesteps - 1 if is_pipeline_first_stage() else num_timesteps
        for _ in range(recv_timesteps):
            for patch_idx in range(get_runtime_state().num_pipeline_patch):
                if not is_pipeline_first_stage():
                    get_pp_group().add_pipeline_recv_task(patch_idx, name="noise_uncond")
                get_pp_group().add_pipeline_recv_task(patch_idx)

        return patch_latents

    def pipefusion_sync_pipeline(
        self,
        timesteps: torch.Tensor,
        latents: torch.Tensor,
        dtype: torch.dtype,
        do_true_cfg_fn: Any,
        guidance_scale_fn: Any,
        sync_only: bool = False,
        **extra_kwargs: Any,
    ) -> torch.Tensor:
        """
        Run the synchronous (warmup) phase of PipeFusion.

        Args:
            timesteps: Timestep schedule for this phase.
            latents: Current latents.
            dtype: Working dtype.
            do_true_cfg_fn: Callable(timestep_index) -> bool, whether CFG is active.
            guidance_scale_fn: Callable(timestep_index) -> float, current guidance scale.
            sync_only: If True, skip final send on last timestep (no async phase follows).
            **extra_kwargs: Passed through to prepare_pipefusion_noise_kwargs.

        Returns:
            Updated latents.
        """
        latents = self._init_sync_pipeline(latents)
        noise_uncond = None

        for i, t in enumerate(timesteps):
            self._current_timestep = t

            do_true_cfg = do_true_cfg_fn(t)
            current_guidance_scale = guidance_scale_fn(t)

            if is_pipeline_last_stage():
                last_timestep_latents = latents

            # when there is only one pp stage, no need to recv
            if get_pipeline_parallel_world_size() == 1:
                pass
            # all ranks should recv the latent from the previous rank except
            #   the first rank in the first pipeline forward which should use
            #   the input latent
            elif is_pipeline_first_stage() and i == 0:
                pass
            else:
                latents = get_pp_group().pipeline_recv()
                if do_true_cfg and not is_pipeline_first_stage():
                    noise_uncond = get_pp_group().pipeline_recv(name="noise_uncond")

            positive_kwargs, negative_kwargs = self.prepare_pipefusion_noise_kwargs(
                latents=latents.to(dtype),
                timestep=t,
                do_true_cfg=do_true_cfg,
                noise_uncond=noise_uncond,
                **extra_kwargs,
            )

            # Predict noise with automatic CFG parallel handling
            noise_pred = self.predict_noise_maybe_with_cfg(
                do_true_cfg=do_true_cfg,
                true_cfg_scale=current_guidance_scale,
                positive_kwargs=positive_kwargs,
                negative_kwargs=negative_kwargs,
                cfg_normalize=False,
            )

            if is_pipeline_last_stage():
                # Compute the previous noisy sample x_t -> x_t-1 with automatic CFG sync
                latents = self.scheduler_step_maybe_with_cfg(noise_pred, t, last_timestep_latents, do_true_cfg)

            if sync_only and is_pipeline_last_stage() and i == len(timesteps) - 1:
                pass
            elif get_pipeline_parallel_world_size() > 1:
                if is_pipeline_last_stage():
                    get_pp_group().pipeline_send(latents.to(dtype))
                else:
                    get_pp_group().pipeline_send(noise_pred[0])
                    if do_true_cfg:
                        get_pp_group().pipeline_send(noise_pred[1], name="noise_uncond")

        return latents

    def pipefusion_async_pipeline(
        self,
        timesteps: torch.Tensor,
        latents: torch.Tensor,
        dtype: torch.dtype,
        do_true_cfg_fn: Any,
        guidance_scale_fn: Any,
        **extra_kwargs: Any,
    ) -> torch.Tensor | None:
        """
        Run the asynchronous (patched) phase of PipeFusion.

        Args:
            timesteps: Timestep schedule for this phase.
            latents: Current latents (from sync phase).
            dtype: Working dtype.
            do_true_cfg_fn: Callable(timestep_index) -> bool.
            guidance_scale_fn: Callable(timestep_index) -> float.
            **extra_kwargs: Passed through to prepare_pipefusion_noise_kwargs.

        Returns:
            Updated latents (only on last pipeline stage), None otherwise.
        """
        if len(timesteps) == 0:
            return latents
        num_pipeline_patch = get_runtime_state().num_pipeline_patch
        patch_latents = self._init_async_pipeline(num_timesteps=len(timesteps), latents=latents)
        last_patch_latents = [None] * num_pipeline_patch if is_pipeline_last_stage() else None
        noise_uncond = None

        first_async_recv = True
        for i, t in enumerate(timesteps):
            self._current_timestep = t

            do_true_cfg = do_true_cfg_fn(t)
            current_guidance_scale = guidance_scale_fn(t)

            for patch_idx in range(num_pipeline_patch):
                if is_pipeline_last_stage():
                    last_patch_latents[patch_idx] = patch_latents[patch_idx]

                if is_pipeline_first_stage() and i == 0:
                    pass
                else:
                    if first_async_recv:
                        if do_true_cfg and not is_pipeline_first_stage():
                            get_pp_group().recv_next()
                        get_pp_group().recv_next()
                        first_async_recv = False

                    if do_true_cfg and not is_pipeline_first_stage():
                        noise_uncond = get_pp_group().get_pipeline_recv_data(idx=patch_idx, name="noise_uncond")
                    patch_latents[patch_idx] = get_pp_group().get_pipeline_recv_data(idx=patch_idx)

                positive_kwargs, negative_kwargs = self.prepare_pipefusion_noise_kwargs(
                    latents=patch_latents[patch_idx].to(dtype),
                    timestep=t,
                    do_true_cfg=do_true_cfg,
                    noise_uncond=noise_uncond,
                    **extra_kwargs,
                )

                # Predict noise with automatic CFG parallel handling
                patch_latents[patch_idx] = self.predict_noise_maybe_with_cfg(
                    do_true_cfg=do_true_cfg,
                    true_cfg_scale=current_guidance_scale,
                    positive_kwargs=positive_kwargs,
                    negative_kwargs=negative_kwargs,
                    cfg_normalize=False,
                )

                if is_pipeline_last_stage():
                    # Compute the previous noisy sample x_t -> x_t-1 with automatic CFG sync
                    patch_latents[patch_idx] = self.scheduler_step_maybe_with_cfg(
                        patch_latents[patch_idx], t, last_patch_latents[patch_idx], do_true_cfg
                    )
                    if i != len(timesteps) - 1:
                        get_pp_group().pipeline_isend(patch_latents[patch_idx].to(dtype), segment_idx=patch_idx)
                else:
                    if do_true_cfg:
                        patch_latents[patch_idx], noise_uncond = patch_latents[patch_idx]
                        get_pp_group().pipeline_isend(noise_uncond, name="noise_uncond", segment_idx=patch_idx)
                    get_pp_group().pipeline_isend(patch_latents[patch_idx], segment_idx=patch_idx)

                if is_pipeline_first_stage() and i == 0:
                    pass
                else:
                    if i == len(timesteps) - 1 and patch_idx == num_pipeline_patch - 1:
                        pass
                    elif is_pipeline_first_stage():
                        get_pp_group().recv_next()
                    else:
                        # recv noise_uncond
                        get_pp_group().recv_next()
                        # recv latents
                        get_pp_group().recv_next()

                get_runtime_state().next_patch()

        latents = None
        if is_pipeline_last_stage():
            latents = torch.cat(patch_latents, dim=get_runtime_state().latent_split_dim)
        return latents

    @staticmethod
    def pipefusion_send_output_to_first_rank(output: torch.Tensor | None) -> torch.Tensor | None:
        """Send final output from last rank to first rank for pipeline parallel."""
        if get_pipeline_parallel_world_size() > 1:
            if is_pipeline_last_stage():
                get_pp_group().send_tensor_dict({"output": output}, dst=0)
            elif is_pipeline_first_stage():
                output_dict = get_pp_group().recv_tensor_dict(src=get_pipeline_parallel_world_size() - 1)
                output = output_dict["output"]
        return output

    def pipefusion_distribute_latents(
        self,
        latents: torch.Tensor,
        is_distributed_vae: bool,
        vae_dtype: torch.dtype,
        vae_device: torch.device,
        original_dims: tuple[int, ...],
    ) -> torch.Tensor:
        """
        Distribute latents to the appropriate rank(s) for VAE decoding.

        If VAE parallel is enabled, broadcast latents from the last pipeline rank to all ranks in the
        distributed VAE group.
        If VAE parallel is disabled, send latents from the last pipeline rank to the first pipeline rank,
        so that decoding can happen on the first rank.
        """
        if is_distributed_vae:
            dit_group = get_dit_group()
            dit_rank = dist.get_rank(dit_group)
            src_rank = dist.get_world_size(dit_group) - 1

            if dit_rank == src_rank:
                latents = latents.to(vae_dtype)
            else:
                latents = torch.empty(original_dims, dtype=vae_dtype, device=vae_device)
            dist.broadcast(latents, src=src_rank, group=dit_group)
        else:
            latents = self.pipefusion_send_output_to_first_rank(latents)
        return latents
