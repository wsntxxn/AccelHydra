import copy
import inspect
import random
from typing import Any, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accel_hydra.models.common import (
    CountParamsBase,
    LoadPretrainedBase,
    SaveTrainableParamsBase,
)
from accel_hydra.utils.torch import (
    create_mask_from_length,
    loss_with_mask,
    trim_or_pad_length,
)
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.training_utils import compute_density_for_timestep_sampling
from diffusers.utils.torch_utils import randn_tensor
from safetensors.torch import load_file
from tqdm import tqdm

from models.autoencoder.autoencoder_base import AutoEncoderBase
from models.content_adapter import ContentAdapterBase
from models.content_encoder.content_encoder import ContentEncoder


class FlowMatchingMixin:
    def __init__(
        self,
        cfg_drop_ratio: float = 0.2,
        sample_strategy: str = 'normal',
        num_train_steps: int = 1000
    ) -> None:
        r"""
        Args:
            cfg_drop_ratio (float): Dropout ratio for the autoencoder.
            sample_strategy (str): Sampling strategy for timesteps during training.
            num_train_steps (int): Number of training steps for the noise scheduler.
        """
        self.sample_strategy = sample_strategy
        self.infer_noise_scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=num_train_steps
        )
        self.train_noise_scheduler = copy.deepcopy(self.infer_noise_scheduler)

        self.classifier_free_guidance = cfg_drop_ratio > 0.0
        self.cfg_drop_ratio = cfg_drop_ratio

    def get_input_target_and_timesteps(
        self, latent: torch.Tensor, training: bool = True
    ):
        bsz = latent.shape[0]
        noise = torch.randn_like(latent)

        if training:
            if self.sample_strategy == 'normal':
                u = compute_density_for_timestep_sampling(
                    weighting_scheme="logit_normal",
                    batch_size=bsz,
                    logit_mean=0,
                    logit_std=1,
                    mode_scale=None,
                )
            elif self.sample_strategy == 'uniform':
                u = torch.randn(bsz, )
            else:
                raise NotImplementedError(
                    f"{self.sample_strategy} samlping for timesteps is not supported now"
                )
        else:
            u = torch.ones(bsz, ) / 2

        indices = (u * self.train_noise_scheduler.config.num_train_timesteps
                  ).long()

        # train_noise_scheduler.timesteps: a list from 1 ~ num_trainsteps with 1 as interval
        timesteps = self.train_noise_scheduler.timesteps[indices].to(
            device=latent.device
        )
        sigmas = self.get_sigmas(
            timesteps, n_dim=latent.ndim, dtype=latent.dtype
        )

        noisy_latent = (1.0 - sigmas) * latent + sigmas * noise

        target = noise - latent

        return noisy_latent, target, timesteps

    def get_sigmas(self, timesteps, n_dim=3, dtype=torch.float32):
        device = timesteps.device

        # a list from 1 declining to 1/num_train_steps
        sigmas = self.train_noise_scheduler.sigmas.to(
            device=device, dtype=dtype
        )

        schedule_timesteps = self.train_noise_scheduler.timesteps.to(device)
        timesteps = timesteps.to(device)
        step_indices = [(schedule_timesteps == t).nonzero().item()
                        for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def retrieve_timesteps(
        self,
        num_inference_steps: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
        timesteps: Optional[List[int]] = None,
        sigmas: Optional[List[float]] = None,
        **kwargs,
    ):
        # used in inference, retrieve new timesteps on given inference timesteps
        scheduler = self.infer_noise_scheduler

        if timesteps is not None and sigmas is not None:
            raise ValueError(
                "Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values"
            )
        if timesteps is not None:
            accepts_timesteps = "timesteps" in set(
                inspect.signature(scheduler.set_timesteps).parameters.keys()
            )
            if not accepts_timesteps:
                raise ValueError(
                    f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                    f" timestep schedules. Please check whether you are using the correct scheduler."
                )
            scheduler.set_timesteps(
                timesteps=timesteps, device=device, **kwargs
            )
            timesteps = scheduler.timesteps
            num_inference_steps = len(timesteps)
        elif sigmas is not None:
            accept_sigmas = "sigmas" in set(
                inspect.signature(scheduler.set_timesteps).parameters.keys()
            )
            if not accept_sigmas:
                raise ValueError(
                    f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                    f" sigmas schedules. Please check whether you are using the correct scheduler."
                )
            scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
            timesteps = scheduler.timesteps
            num_inference_steps = len(timesteps)
        else:
            scheduler.set_timesteps(
                num_inference_steps, device=device, **kwargs
            )
            timesteps = scheduler.timesteps
        return timesteps, num_inference_steps


class ContentEncoderAdapterMixin:
    def __init__(
        self,
        content_encoder: ContentEncoder,
        content_adapter: ContentAdapterBase | None = None
    ):
        self.content_encoder = content_encoder
        self.content_adapter = content_adapter

    def encode_content(
        self,
        content: list[Any],
        task: list[str],
        device: str | torch.device,
        instruction: torch.Tensor | None = None,
        instruction_lengths: torch.Tensor | None = None
    ):
        content_output: dict[
            str, torch.Tensor] = self.content_encoder.encode_content(
                content, task, device=device
            )
        content, content_mask = content_output["content"], content_output[
            "content_mask"]

        if instruction is not None:
            instruction_mask = create_mask_from_length(instruction_lengths)
            (
                content,
                content_mask,
                global_duration_pred,
                local_duration_pred,
            ) = self.content_adapter(
                content, content_mask, instruction, instruction_mask
            )

        return_dict = {
            "content": content,
            "content_mask": content_mask,
            "length_aligned_content": content_output["length_aligned_content"],
        }
        if instruction is not None:
            return_dict["global_duration_pred"] = global_duration_pred
            return_dict["local_duration_pred"] = local_duration_pred

        return return_dict


class SingleTaskCrossAttentionAudioFlowMatching(
    LoadPretrainedBase, CountParamsBase, SaveTrainableParamsBase,
    FlowMatchingMixin, ContentEncoderAdapterMixin
):
    def __init__(
        self,
        autoencoder: nn.Module,
        content_encoder: ContentEncoder,
        backbone: nn.Module,
        cfg_drop_ratio: float = 0.2,
        sample_strategy: str = 'normal',
        num_train_steps: int = 1000,
        pretrained_ckpt: str | None = None,
    ):
        nn.Module.__init__(self)
        FlowMatchingMixin.__init__(
            self, cfg_drop_ratio, sample_strategy, num_train_steps
        )
        ContentEncoderAdapterMixin.__init__(
            self, content_encoder=content_encoder
        )

        self.autoencoder = autoencoder
        for param in self.autoencoder.parameters():
            param.requires_grad = False

        if hasattr(self.content_encoder, "audio_encoder"):
            if self.content_encoder.audio_encoder is not None:
                self.content_encoder.audio_encoder.model = self.autoencoder

        self.backbone = backbone
        self.dummy_param = nn.Parameter(torch.empty(0))

        if pretrained_ckpt is not None:
            print(f"Load pretrain FlowMatching model from {pretrained_ckpt}")
            pretrained_state_dict = load_file(pretrained_ckpt)
            self.load_pretrained(pretrained_state_dict)
            # missing, unexpected = self.load_state_dict(pretrained_state_dict, strict=False)
            # print("Missing keys:", missing)
            # print("Unexpected keys:", unexpected)

        # if content_encoder.embed_dim != 1024:
        #     self.context_proj = nn.Sequential(
        #         nn.Linear(content_encoder.embed_dim, 1024),
        #         nn.SiLU(),
        #         nn.Linear(1024, 1024),
        #     )
        # else:
        #     self.context_proj = nn.Identity()

    def forward(
        self,
        content: list[Any],
        condition: list[Any],
        task: list[str],
        waveform: torch.Tensor,
        waveform_lengths: torch.Tensor,
        loss_reduce: bool = True,
        **kwargs
    ):
        loss_reduce = self.training or (loss_reduce and not self.training)
        device = self.dummy_param.device

        self.autoencoder.eval()
        with torch.no_grad():
            latent, latent_mask = self.autoencoder.encode(
                waveform.unsqueeze(1), waveform_lengths
            )

        content_dict = self.encode_content(content, task, device)
        content, content_mask = content_dict["content"], content_dict[
            "content_mask"]

        # content = self.context_proj(content)

        if self.training and self.classifier_free_guidance:
            mask_indices = [
                k for k in range(len(waveform))
                if random.random() < self.cfg_drop_ratio
            ]
            if len(mask_indices) > 0:
                content[mask_indices] = 0

        noisy_latent, target, timesteps = self.get_input_target_and_timesteps(
            latent, training=self.training
        )

        pred: torch.Tensor = self.backbone(
            x=noisy_latent,
            timesteps=timesteps,
            context=content,
            x_mask=latent_mask,
            context_mask=content_mask
        )

        diff_loss = F.mse_loss(pred.float(), target.float(), reduction="none")
        diff_loss = loss_with_mask(
            diff_loss, latent_mask.unsqueeze(1), reduce=loss_reduce
        )
        #diff_loss = loss_with_mask(diff_loss, latent_mask.unsqueeze(1))
        output = {"diff_loss": diff_loss}
        return output

    def iterative_denoise(
        self, latent: torch.Tensor, timesteps: list[int], num_steps: int,
        verbose: bool, cfg: bool, cfg_scale: float, backbone_input: dict
    ):
        progress_bar = tqdm(range(num_steps), disable=not verbose)

        for i, timestep in enumerate(timesteps):
            # expand the latent if we are doing classifier free guidance
            if cfg:
                latent_input = torch.cat([latent, latent])
            else:
                latent_input = latent

            noise_pred: torch.Tensor = self.backbone(
                x=latent_input, timesteps=timestep, **backbone_input
            )

            # perform guidance
            if cfg:
                noise_pred_uncond, noise_pred_content = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + cfg_scale * (
                    noise_pred_content - noise_pred_uncond
                )

            latent = self.infer_noise_scheduler.step(
                noise_pred, timestep, latent
            ).prev_sample

            progress_bar.update(1)

        progress_bar.close()

        return latent

    @torch.no_grad()
    def inference(
        self,
        content: list[Any],
        condition: list[Any],
        task: list[str],
        latent_shape: Sequence[int],
        num_steps: int = 50,
        sway_sampling_coef: float | None = -1.0,
        guidance_scale: float = 3.0,
        num_samples_per_content: int = 1,
        disable_progress: bool = True,
        **kwargs
    ):
        device = self.dummy_param.device
        classifier_free_guidance = guidance_scale > 1.0
        batch_size = len(content) * num_samples_per_content

        if classifier_free_guidance:
            content, content_mask = self.encode_content_classifier_free(
                content, task, device, num_samples_per_content
            )
        else:
            content_output: dict[
                str, torch.Tensor] = self.content_encoder.encode_content(
                    content, task
                )
            content, content_mask = content_output["content"], content_output[
                "content_mask"]
            content = content.repeat_interleave(num_samples_per_content, 0)
            content_mask = content_mask.repeat_interleave(
                num_samples_per_content, 0
            )

        latent = self.prepare_latent(
            batch_size, latent_shape, content.dtype, device
        )

        if not sway_sampling_coef:
            sigmas = np.linspace(1.0, 1 / num_steps, num_steps)
        else:
            t = torch.linspace(0, 1, num_steps + 1)
            t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)
            sigmas = 1 - t
        timesteps, num_steps = self.retrieve_timesteps(
            num_steps, device, timesteps=None, sigmas=sigmas
        )

        latent = self.iterative_denoise(
            latent=latent,
            timesteps=timesteps,
            num_steps=num_steps,
            verbose=not disable_progress,
            cfg=classifier_free_guidance,
            cfg_scale=guidance_scale,
            backbone_input={
                "context": content,
                "context_mask": content_mask,
            },
        )

        waveform = self.autoencoder.decode(latent)

        return waveform

    def prepare_latent(
        self, batch_size: int, latent_shape: Sequence[int], dtype: torch.dtype,
        device: str
    ):
        shape = (batch_size, *latent_shape)
        latent = randn_tensor(
            shape, generator=None, device=device, dtype=dtype
        )
        return latent

    def encode_content_classifier_free(
        self,
        content: list[Any],
        task: list[str],
        device,
        num_samples_per_content: int = 1
    ):
        content_dict = self.content_encoder.encode_content(
            content, task, device
        )
        content, content_mask = content_dict["content"], content_dict[
            "content_mask"]
        # content, content_mask = self.content_encoder.encode_content(
        #     content, task, device=device
        # )

        content = content.repeat_interleave(num_samples_per_content, 0)
        content_mask = content_mask.repeat_interleave(
            num_samples_per_content, 0
        )

        # get unconditional embeddings for classifier free guidance
        uncond_content = torch.zeros_like(content)
        uncond_content_mask = content_mask.detach().clone()

        uncond_content = uncond_content.repeat_interleave(
            num_samples_per_content, 0
        )
        uncond_content_mask = uncond_content_mask.repeat_interleave(
            num_samples_per_content, 0
        )

        # For classifier free guidance, we need to do two forward passes.
        # We concatenate the unconditional and text embeddings into a single batch to avoid doing two forward passes
        content = torch.cat([uncond_content, content])
        content_mask = torch.cat([uncond_content_mask, content_mask])

        return content, content_mask


class MultiContentAudioFlowMatching(SingleTaskCrossAttentionAudioFlowMatching):
    def __init__(
        self,
        autoencoder: AutoEncoderBase,
        content_encoder: ContentEncoder,
        backbone: nn.Module,
        cfg_drop_ratio: float = 0.2,
        sample_strategy: str = 'normal',
        num_train_steps: int = 1000,
        pretrained_ckpt: str | None = None,
        embed_dim: int = 1024,
    ):
        super().__init__(
            autoencoder=autoencoder,
            content_encoder=content_encoder,
            backbone=backbone,
            cfg_drop_ratio=cfg_drop_ratio,
            sample_strategy=sample_strategy,
            num_train_steps=num_train_steps,
            pretrained_ckpt=pretrained_ckpt,
        )

    def forward(
        self,
        content: list[Any],
        duration: Sequence[float],
        task: list[str],
        waveform: torch.Tensor,
        waveform_lengths: torch.Tensor,
        loss_reduce: bool = True,
        **kwargs
    ):
        device = self.dummy_param.device
        loss_reduce = self.training or (loss_reduce and not self.training)

        self.autoencoder.eval()

        with torch.no_grad():
            latent, latent_mask = self.autoencoder.encode(
                waveform.unsqueeze(1), waveform_lengths
            )  # latent [B, 128, 500/T=10s], latent_mask [B, 500/T=10s]

        content_dict = self.encode_content(content, task, device)
        context, context_mask, length_aligned_content = content_dict[
            "content"], content_dict["content_mask"], content_dict[
                "length_aligned_content"]

        # --------------------------------------------------------------------
        # prepare latent and noise
        # --------------------------------------------------------------------
        noisy_latent, target, timesteps = self.get_input_target_and_timesteps(
            latent, training=self.training
        )

        # --------------------------------------------------------------------
        # prepare input to the backbone
        # --------------------------------------------------------------------
        # TODO compatility for 2D spectrogram VAE

        latent_length = noisy_latent.size(self.autoencoder.time_dim)
        time_aligned_content = trim_or_pad_length(
            length_aligned_content, latent_length, 1
        )

        # --------------------------------------------------------------------
        # classifier free guidance
        # --------------------------------------------------------------------
        if self.training and self.classifier_free_guidance:
            mask_indices = [
                k for k in range(len(waveform))
                if random.random() < self.cfg_drop_ratio
            ]
            if len(mask_indices) > 0:
                context[mask_indices] = 0
                time_aligned_content[mask_indices] = 0

        pred: torch.Tensor = self.backbone(
            x=noisy_latent,
            x_mask=latent_mask,
            timesteps=timesteps,
            context=context,
            context_mask=context_mask,
            time_aligned_context=time_aligned_content,
        )

        pred = pred.transpose(1, self.autoencoder.time_dim)
        target = target.transpose(1, self.autoencoder.time_dim)
        diff_loss = F.mse_loss(pred.float(), target.float(), reduction="none")
        diff_loss = loss_with_mask(diff_loss, latent_mask, reduce=loss_reduce)

        return {
            "diff_loss": diff_loss,
        }

    def inference(
        self,
        content: list[Any],
        task: list[str],
        latent_shape: Sequence[int],
        num_steps: int = 50,
        sway_sampling_coef: float | None = -1.0,
        guidance_scale: float = 3.0,
        disable_progress: bool = True,
        **kwargs
    ):
        device = self.dummy_param.device
        classifier_free_guidance = guidance_scale > 1.0
        batch_size = len(content)

        content_dict: dict[str, torch.Tensor] = self.encode_content(
            content, task, device
        )
        context, context_mask, length_aligned_content = \
            content_dict["content"], content_dict[
                "content_mask"],  content_dict["length_aligned_content"]

        shape = (batch_size, *latent_shape)
        latent_length = shape[self.autoencoder.time_dim]
        time_aligned_content = trim_or_pad_length(
            length_aligned_content, latent_length, 1
        )

        # --------------------------------------------------------------------
        # prepare unconditional input
        # --------------------------------------------------------------------
        if classifier_free_guidance:
            uncond_time_aligned_content = torch.zeros_like(
                time_aligned_content
            )
            uncond_context = torch.zeros_like(context)
            uncond_context_mask = context_mask.detach().clone()
            time_aligned_content = torch.cat([
                uncond_time_aligned_content, time_aligned_content
            ])
            context = torch.cat([uncond_context, context])
            context_mask = torch.cat([uncond_context_mask, context_mask])

        latent = randn_tensor(
            shape, generator=None, device=device, dtype=context.dtype
        )

        if not sway_sampling_coef:
            sigmas = np.linspace(1.0, 1 / num_steps, num_steps)
        else:
            t = torch.linspace(0, 1, num_steps + 1)
            t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)
            sigmas = 1 - t
        timesteps, num_steps = self.retrieve_timesteps(
            num_steps, device, timesteps=None, sigmas=sigmas
        )
        latent = self.iterative_denoise(
            latent=latent,
            timesteps=timesteps,
            num_steps=num_steps,
            verbose=not disable_progress,
            cfg=classifier_free_guidance,
            cfg_scale=guidance_scale,
            backbone_input={
                "context": context,
                "context_mask": context_mask,
                "time_aligned_context": time_aligned_content,
            }
        )

        waveform = self.autoencoder.decode(latent)
        return waveform
