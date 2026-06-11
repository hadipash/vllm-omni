# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project and the xDiT authors.
#
# This module is adapted from xDiT (https://github.com/xdit-project/xdit)

"""
PipeFusion mixin for diffusion pipelines.

Provides patch-wise pipeline parallelism for diffusion models.
The warmup phase uses the standard denoising loop (via PipelineParallelMixin),
and the async phase splits latents into patches and processes each patch
through predict_noise_maybe_with_cfg + scheduler_step_maybe_with_cfg.
"""

import inspect
from abc import ABC, abstractmethod
from functools import wraps
from typing import TYPE_CHECKING, Any

import torch
from vllm.sequence import IntermediateTensors

from vllm_omni.diffusion.distributed.parallel_state import (
    get_classifier_free_guidance_world_size,
    get_pipeline_parallel_rank,
    is_pipeline_first_stage,
    is_pipeline_intermediate_stage,
    is_pipeline_last_stage,
)
from vllm_omni.diffusion.distributed.pipefusion.pipefusion_runtime import (
    get_pipefusion_runtime,
    is_pipefusion_initialized,
)
from vllm_omni.diffusion.forward_context import set_forward_context_denoise_step_idx

if TYPE_CHECKING:
    from vllm_omni.diffusion.request import OmniDiffusionRequest


class PipeFusionPipelineMixin(ABC):
    """
    Mixin class providing PipeFusion (patch-wise + pipeline parallel) logic
    for diffusion pipelines.

    Required methods (to be implemented by subclasses):
        - predict_noise_maybe_with_cfg(): From PipelineParallelMixin.
        - scheduler_step_maybe_with_cfg(): From PipelineParallelMixin.
        - prepare_model_kwargs(): Returns (positive_kwargs, negative_kwargs) for the current timestep/patch.
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        from vllm_omni.diffusion.distributed.pipeline_parallel import PipelineParallelMixin

        if not issubclass(cls, PipelineParallelMixin):
            raise TypeError(
                f"{cls.__name__} inherits PipeFusionPipelineMixin but not PipelineParallelMixin. "
                "PipeFusion requires PipelineParallelMixin for _sync_pp_send(), "
                "predict_noise_maybe_with_cfg(), and scheduler_step_maybe_with_cfg(). "
                "Add PipelineParallelMixin to the base classes of your pipeline."
            )

        mro = cls.mro()
        if mro.index(PipeFusionPipelineMixin) > mro.index(PipelineParallelMixin):
            raise TypeError(
                f"{cls.__name__} must inherit PipeFusionPipelineMixin before "
                "PipelineParallelMixin so MRO selects patch-aware predict/scheduler wrappers"
            )

        if is_pipefusion_initialized():
            runtime = get_pipefusion_runtime()

            init = cls.__dict__.get("__init__")
            if callable(init):

                @wraps(init)
                def wrapped_init(self, *args: Any, **kwargs: Any) -> None:
                    init(self, *args, **kwargs)
                    # Initialize the patch size to transformer's patch size
                    runtime.patch_size = self.transformer_config.patch_size

                cls.__init__ = wrapped_init

            forward = cls.__dict__.get("forward")
            if callable(forward):

                @wraps(forward)
                def wrapped_forward(self, req: "OmniDiffusionRequest", *args: Any, **kwargs: Any) -> Any:
                    # capture the number of warm-up steps and split dimension
                    self._configure_pipefusion_run(req)
                    return forward(self, req, *args, **kwargs)

                cls.forward = wrapped_forward

            diffuse = cls.__dict__.get("diffuse")
            if callable(diffuse):
                sig = inspect.signature(diffuse)

                @wraps(diffuse)
                def wrapped_diffuse(self, *args: Any, **kwargs: Any) -> Any:
                    bound = sig.bind(self, *args, **kwargs)
                    bound.apply_defaults()

                    latents = bound.arguments["latents"]
                    timesteps = bound.arguments["timesteps"]
                    dtype = bound.arguments["dtype"]

                    # PipeFusion: set runtime state input parameters and reset caches
                    runtime.set_input_parameters(latents, dtype)
                    self.scheduler.clear_patch_caches()
                    self._reset_pipefusion_caches()

                    warmup_steps = runtime.warmup_steps
                    warmup_timesteps = timesteps[:warmup_steps]
                    async_timesteps = timesteps[warmup_steps:] if len(timesteps) > warmup_steps else None
                    runtime.warmup_cache_timestep = warmup_timesteps[-1]

                    # Call standard diffuse for warmup steps
                    bound.arguments["timesteps"] = warmup_timesteps
                    latents = diffuse(*bound.args, **bound.kwargs)

                    if async_timesteps is not None:
                        with self.progress_bar(total=len(async_timesteps)) as pbar:
                            latents = self._async_pipeline(timesteps=async_timesteps, latents=latents, pbar=pbar)

                    return latents

                cls.diffuse = wrapped_diffuse

            prepare_model_kwargs = cls.__dict__.get("prepare_model_kwargs")
            if callable(prepare_model_kwargs):

                @wraps(prepare_model_kwargs)
                def wrapped_prepare_model_kwargs(self, latents: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
                    # Only the first PP stage consumes latents; later stages receive intermediate tensors.
                    if is_pipeline_first_stage():
                        return prepare_model_kwargs(self, latents, *args, **kwargs)
                    return prepare_model_kwargs(self, None, *args, **kwargs)

                cls.prepare_model_kwargs = wrapped_prepare_model_kwargs

            predict_noise_maybe_with_cfg = getattr(cls, "predict_noise_maybe_with_cfg", None)
            if callable(predict_noise_maybe_with_cfg):

                @wraps(predict_noise_maybe_with_cfg)
                def wrapped_predict_noise_maybe_with_cfg(self, *args: Any, **kwargs: Any) -> Any:
                    # Update the KV cache only during the final warmup step
                    runtime.update_warmup_cache = not runtime.patch_mode and torch.equal(
                        self._current_timestep, runtime.warmup_cache_timestep
                    )
                    return predict_noise_maybe_with_cfg(self, *args, **kwargs)

                cls.predict_noise_maybe_with_cfg = wrapped_predict_noise_maybe_with_cfg

            predict_noise = getattr(cls, "predict_noise", None)
            if callable(predict_noise):

                @wraps(predict_noise)
                def wrapped_predict_noise(self, *args: Any, **kwargs: Any) -> Any:
                    if (intermediate_tensors := kwargs.get("intermediate_tensors")) is not None:
                        if (
                            self._pipefusion_capture_intermediate_tensors
                            and runtime.pipeline_patch_idx == self._pipefusion_capture_patch_idx
                            and not is_pipeline_first_stage()
                        ):
                            self._pipefusion_warmup_intermediate_tensors.append(intermediate_tensors)
                        if self._pipefusion_capture_last_stage_intermediate_tensors:
                            self._pipefusion_last_stage_intermediate_tensors.append(intermediate_tensors)
                    return predict_noise(self, *args, **kwargs)

                cls.predict_noise = wrapped_predict_noise

    @staticmethod
    def _configure_pipefusion_run(req: "OmniDiffusionRequest") -> None:
        if sampling_params := getattr(req, "sampling_params", None):
            get_pipefusion_runtime().set_run_config(
                warmup_steps=getattr(sampling_params, "pipefusion_warmup_steps", None),
                split_dim=getattr(sampling_params, "pipefusion_split_dim", None),
                use_rotational_pipefusion=getattr(sampling_params, "enable_rotational_pipefusion", None),
            )

    def _reset_pipefusion_caches(self) -> None:
        self._pipefusion_warmup_intermediate_tensors: list[IntermediateTensors] = []
        self._pipefusion_last_stage_intermediate_tensors: list[IntermediateTensors] = []
        self._pipefusion_capture_intermediate_tensors = False
        self._pipefusion_capture_last_stage_intermediate_tensors = False
        self._pipefusion_capture_patch_idx = 0
        for module in self.modules():
            if callable(reset_cache := getattr(module, "pipefusion_reset_cache", None)):
                reset_cache()

    @abstractmethod
    def prepare_model_kwargs(
        self,
        latents: torch.Tensor | None,
        timestep: torch.Tensor,
        **extra_kwargs: Any,
    ) -> tuple[dict[str, Any], dict[str, Any] | None, bool, float]:
        """
        Prepare positive and negative kwargs for noise prediction.

        Subclasses MUST implement this to provide model-specific kwargs.

        Args:
            latents: The input latents for this step/patch.
            timestep: Current timestep tensor.
            **extra_kwargs: Additional pipeline-specific arguments.

        Returns:
            (positive_kwargs, negative_kwargs, do_true_cfg, scale) for predict_noise_maybe_with_cfg.
        """
        raise NotImplementedError("Subclasses must implement prepare_model_kwargs")

    def _async_pipeline(self, timesteps: torch.Tensor, latents: torch.Tensor, pbar) -> torch.Tensor | None:
        """
        Run the asynchronous (patched) phase of PipeFusion.

        Splits latents into spatial patches and runs the standard
        predict_noise_maybe_with_cfg + scheduler_step_maybe_with_cfg
        loop for each patch, using PipelineParallelMixin for all inter-stage
        communication.

        Args:
            timesteps: Timestep schedule for this phase.
            latents: Current latents (from warmup phase).

        Returns:
            Updated latents on the first pipeline stage, None otherwise.
        """
        runtime = get_pipefusion_runtime()
        runtime.set_patched_mode(patch_mode=True)

        # Split latents into patches
        split_sizes = runtime.pp_patches_height
        split_dim = runtime.latent_split_dim
        patch_latents = list(latents.split(split_sizes, dim=split_dim))
        if is_pipeline_last_stage():
            # Split scheduler caches into per-patch versions for async pipeline
            self.scheduler.split_caches_for_patches(split_sizes, dim=split_dim)

        num_patch = runtime.num_pipeline_patch
        patch_indices = list(range(num_patch))
        rotation_requested = runtime.use_rotational_pipefusion
        if rotation_requested:
            self._pipefusion_capture_patch_idx = runtime.get_initial_patch_indices()[0]

        for i, t in enumerate(timesteps):
            pp_rank = get_pipeline_parallel_rank()
            self._current_timestep = t
            set_forward_context_denoise_step_idx(runtime.warmup_steps + i)
            self._pipefusion_capture_intermediate_tensors = rotation_requested and i == 0
            self._pipefusion_capture_last_stage_intermediate_tensors = False

            # First async timestep runs in original PipeFusion order to warm
            # per-comm-id metadata before rotational out-of-order receives.
            rotation_active = rotation_requested and i > 0
            if rotation_active and i == 1:
                patch_indices = runtime.get_initial_patch_indices()
            # in Rotational PipeFusion, `patch_latents` stores latents for the next timestep
            patch_latents_step = patch_latents.copy()
            last_stage_patch_indices = (
                runtime.get_last_stage_patch_indices(i - 1) if rotation_active else patch_indices
            )

            for ip, pidx in enumerate(patch_indices):
                is_first_patch, is_last_patch = (ip == 0), (ip == num_patch - 1)
                skip_patch, skip_recv = False, False
                if rotation_active:
                    skip_patch = is_last_patch
                    if i == len(timesteps) - 1 and not is_pipeline_last_stage() and ip >= pp_rank + 1:
                        skip_patch = True
                    skip_recv = (is_pipeline_intermediate_stage() and i == 1 and is_first_patch) or (
                        is_pipeline_last_stage() and is_first_patch
                    )

                if skip_patch and is_pipeline_intermediate_stage():
                    # intermediate stages can be skipped safely
                    continue

                runtime.next_patch(pidx, is_first_patch=is_first_patch, is_last_patch=is_last_patch)
                if rotation_active and is_pipeline_last_stage() and is_last_patch:
                    self._pipefusion_last_stage_intermediate_tensors = []
                    self._pipefusion_capture_last_stage_intermediate_tensors = True
                else:
                    self._pipefusion_capture_last_stage_intermediate_tensors = False

                positive_kwargs, negative_kwargs, do_true_cfg, scale = self.prepare_model_kwargs(
                    patch_latents_step[pidx], t, **self._pipeline_kwargs
                )

                cfg_parallel_ready = do_true_cfg and get_classifier_free_guidance_world_size() > 1
                n_branches = 1 if (cfg_parallel_ready or not do_true_cfg) else 2

                noise_pred = None
                cached_intermediate_tensors = (
                    self._pipefusion_warmup_intermediate_tensors
                    if not is_pipeline_last_stage() or i == 1
                    else self._pipefusion_last_stage_intermediate_tensors
                ) if skip_recv else None
                if not (skip_patch and is_pipeline_first_stage()):
                    noise_pred = self.predict_noise_maybe_with_cfg(
                        do_true_cfg=do_true_cfg,
                        true_cfg_scale=scale,
                        positive_kwargs=positive_kwargs,
                        negative_kwargs=negative_kwargs,
                        cfg_normalize=False,
                        skip_sync=True,
                        inter_comm_ids=[f"pf-it-{pidx}-{b}" for b in range(n_branches)],
                        intermediate_tensors=cached_intermediate_tensors,
                    )

                if rotation_active:
                    pidx = last_stage_patch_indices[ip]
                updated_latents = self.scheduler_step_maybe_with_cfg(
                    noise_pred,
                    t,
                    patch_latents_step[pidx],
                    do_true_cfg,
                    loopback_comm_id=f"pf-lb-{pidx}",
                )
                if updated_latents is not None:
                    patch_latents[pidx] = updated_latents

            if rotation_active:  # Roll right by 1 for the next timestep
                patch_indices = patch_indices[-1:] + patch_indices[:-1]

            pbar.update()

        # Reassemble latents from patches
        latents = torch.cat(patch_latents, dim=runtime.latent_split_dim)

        # Drain outstanding non-blocking sends before any downstream
        # consumer (e.g. VAE decode broadcast) reuses the buffers.
        self._sync_pp_send()

        runtime.set_patched_mode(patch_mode=False)
        return latents
