"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""
# ruff: noqa: F722 F821

from __future__ import annotations

from random import random
from typing import Callable
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torchdiffeq import odeint
from diffusers.utils.torch_utils import randn_tensor
# from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.training_utils import compute_density_for_timestep_sampling
# from tqdm import tqdm
from accel_hydra.utils.torch import create_mask_from_length as lens_to_mask
from accel_hydra.models.common import CountParamsBase, LoadPretrainedBase

from utils.audio import MelSpec
from utils.general import (
    default,
    exists,
)
from utils.torch import list_str_to_idx, list_str_to_tensor, mask_from_frac_lengths
from utils.flow_matching import get_epss_timesteps
from utils.tokenize import get_tokenizer


class CFM(CountParamsBase, LoadPretrainedBase):
    def __init__(
        self,
        transformer: nn.Module,
        tokenizer_path: str,
        tokenizer: str = "char",
        sigma=0.0,
        odeint_kwargs: dict = dict(
            # atol = 1e-5,
            # rtol = 1e-5,
            method="euler"  # 'midpoint'
        ),
        audio_drop_prob=0.3,
        cond_drop_prob=0.2,
        num_channels=None,
        mel_spec_module: nn.Module | None = None,
        mel_spec_kwargs: dict = dict(),
        frac_lengths_mask: tuple[float, float] = (0.7, 1.0),
        # vocab_char_map: dict[str:int] | None = None,
    ):
        super().__init__()

        self.frac_lengths_mask = frac_lengths_mask

        # mel spec
        self.mel_spec = default(mel_spec_module, MelSpec(**mel_spec_kwargs))
        num_channels = default(num_channels, self.mel_spec.n_mel_channels)
        self.num_channels = num_channels

        # classifier-free guidance
        self.audio_drop_prob = audio_drop_prob
        self.cond_drop_prob = cond_drop_prob

        # transformer
        self.transformer = transformer
        dim = transformer.dim
        self.dim = dim

        # conditional flow related
        self.sigma = sigma

        # sampling related
        self.odeint_kwargs = odeint_kwargs

        # vocab map for tokenization
        # self.vocab_char_map = vocab_char_map
        self.vocab_char_map = get_tokenizer(tokenizer_path, tokenizer)[0]

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def sample(
        self,
        cond: "float['b n d'] | float['b nw']",
        text: "int['b nt'] | list[str]",
        duration: "int | int['b']",
        *,
        lens: "int['b'] | None" = None,
        steps=32,
        cfg_strength=1.0,
        sway_sampling_coef=None,
        seed: int | None = None,
        max_duration=4096,
        vocoder: "Callable[[float['b d n']], float['b nw']] | None" = None,
        use_epss=True,
        no_ref_audio=False,
        duplicate_test=False,
        t_inter=0.1,
        edit_mask=None,
    ):
        self.eval()
        # raw wave

        if cond.ndim == 2:
            cond = self.mel_spec(cond)
            cond = cond.permute(0, 2, 1)
            assert cond.shape[-1] == self.num_channels

        cond = cond.to(next(self.parameters()).dtype)

        batch, cond_seq_len, device = *cond.shape[:2], cond.device
        if not exists(lens):
            lens = torch.full((batch, ),
                              cond_seq_len,
                              device=device,
                              dtype=torch.long)

        # text

        if isinstance(text, list):
            if exists(self.vocab_char_map):
                text = list_str_to_idx(text, self.vocab_char_map).to(device)
            else:
                text = list_str_to_tensor(text).to(device)
            assert text.shape[0] == batch

        # duration

        cond_mask = lens_to_mask(lens)
        if edit_mask is not None:
            cond_mask = cond_mask & edit_mask

        if isinstance(duration, int):
            duration = torch.full((batch, ),
                                  duration,
                                  device=device,
                                  dtype=torch.long)

        duration = torch.maximum(
            torch.maximum((text != -1).sum(dim=-1), lens) + 1, duration
        )  # duration at least text/audio prompt length plus one token, so something is generated
        duration = duration.clamp(max=max_duration)
        max_duration = duration.amax()

        # duplicate test corner for inner time step oberservation
        if duplicate_test:
            test_cond = F.pad(
                cond, (0, 0, cond_seq_len, max_duration - 2 * cond_seq_len),
                value=0.0
            )

        cond = F.pad(cond, (0, 0, 0, max_duration - cond_seq_len), value=0.0)
        if no_ref_audio:
            cond = torch.zeros_like(cond)

        cond_mask = F.pad(
            cond_mask, (0, max_duration - cond_mask.shape[-1]), value=False
        )
        cond_mask = cond_mask.unsqueeze(-1)
        step_cond = torch.where(
            cond_mask, cond, torch.zeros_like(cond)
        )  # allow direct control (cut cond audio) with lens passed in

        if batch > 1:
            mask = lens_to_mask(duration)
        else:  # save memory and speed up, as single inference need no mask currently
            mask = None

        # neural ode

        def fn(t, x):
            # at each step, conditioning is fixed
            # step_cond = torch.where(cond_mask, cond, torch.zeros_like(cond))

            # predict flow (cond)
            if cfg_strength < 1e-5:
                pred = self.transformer(
                    x=x,
                    cond=step_cond,
                    text=text,
                    time=t,
                    mask=mask,
                    drop_audio_cond=False,
                    drop_text=False,
                    cache=True,
                )
                return pred

            # predict flow (cond and uncond), for classifier-free guidance
            pred_cfg = self.transformer(
                x=x,
                cond=step_cond,
                text=text,
                time=t,
                mask=mask,
                cfg_infer=True,
                cache=True,
            )
            pred, null_pred = torch.chunk(pred_cfg, 2, dim=0)
            return pred + (pred - null_pred) * cfg_strength

        # noise input
        # to make sure batch inference result is same with different batch size, and for sure single inference
        # still some difference maybe due to convolutional layers
        y0 = []
        for dur in duration:
            if exists(seed):
                torch.manual_seed(seed)
            y0.append(
                torch.randn(
                    dur,
                    self.num_channels,
                    device=self.device,
                    dtype=step_cond.dtype
                )
            )
        y0 = pad_sequence(y0, padding_value=0, batch_first=True)

        t_start = 0

        # duplicate test corner for inner time step oberservation
        if duplicate_test:
            t_start = t_inter
            y0 = (1 - t_start) * y0 + t_start * test_cond
            steps = int(steps * (1 - t_start))

        if t_start == 0 and use_epss:  # use Empirically Pruned Step Sampling for low NFE
            t = get_epss_timesteps(
                steps, device=self.device, dtype=step_cond.dtype
            )
        else:
            t = torch.linspace(
                t_start,
                1,
                steps + 1,
                device=self.device,
                dtype=step_cond.dtype
            )
        if sway_sampling_coef is not None:
            t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)

        trajectory = odeint(fn, y0, t, **self.odeint_kwargs)
        self.transformer.clear_cache()

        sampled = trajectory[-1]
        out = sampled
        out = torch.where(cond_mask, cond, out)

        if exists(vocoder):
            out = out.permute(0, 2, 1)
            out = vocoder(out)

        return out, trajectory

    def forward(
        self,
        inp: "float['b n d'] | float['b nw']",  # mel or raw wave
        text: "int['b nt'] | list[str]",
        *,
        lens: "int['b'] | None" = None,
        noise_scheduler: str | None = None,
        **kwargs,
    ):
        # handle raw wave
        if inp.ndim == 2:
            inp = self.mel_spec(inp)
            inp = inp.permute(0, 2, 1)
            assert inp.shape[-1] == self.num_channels

        batch, seq_len, dtype, device, _σ1 = *inp.shape[:2
                                                       ], inp.dtype, self.device, self.sigma

        # handle text as string
        if isinstance(text, list):
            if exists(self.vocab_char_map):
                text = list_str_to_idx(text, self.vocab_char_map).to(device)
            else:
                text = list_str_to_tensor(text).to(device)
            assert text.shape[0] == batch

        # lens and mask
        if not exists(lens):  # if lens not acquired by trainer from collate_fn
            lens = torch.full((batch, ), seq_len, device=device)
        mask = lens_to_mask(lens, max_length=seq_len)

        # get a random span to mask out for training conditionally
        frac_lengths = torch.zeros((batch, ),
                                   device=self.device).float().uniform_(
                                       *self.frac_lengths_mask
                                   )
        rand_span_mask = mask_from_frac_lengths(lens, frac_lengths)

        if exists(mask):
            rand_span_mask &= mask

        # mel is x1
        x1 = inp

        # x0 is gaussian noise
        x0 = torch.randn_like(x1)

        # time step
        time = torch.rand((batch, ), dtype=dtype, device=self.device)
        # TODO. noise_scheduler

        # sample xt (φ_t(x) in the paper)
        t = time.unsqueeze(-1).unsqueeze(-1)
        φ = (1 - t) * x0 + t * x1
        flow = x1 - x0

        # only predict what is within the random mask span for infilling
        cond = torch.where(rand_span_mask[..., None], torch.zeros_like(x1), x1)

        # transformer and cfg training with a drop rate
        drop_audio_cond = random(
        ) < self.audio_drop_prob  # p_drop in voicebox paper
        if random() < self.cond_drop_prob:  # p_uncond in voicebox paper
            drop_audio_cond = True
            drop_text = True
        else:
            drop_text = False

        # apply mask will use more memory; might adjust batchsize or batchsampler long sequence threshold
        pred = self.transformer(
            x=φ,
            cond=cond,
            text=text,
            time=time,
            drop_audio_cond=drop_audio_cond,
            drop_text=drop_text,
            mask=mask
        )

        # flow matching loss
        loss = F.mse_loss(pred, flow, reduction="none")
        loss = loss[rand_span_mask]

        # return loss.mean(), cond, pred
        return loss.mean()


class FlowMatchingMixin:
    def __init__(
        self,
        cfg_drop_ratio: float = 0.2,
        sample_strategy: str = 'uniform',
        num_train_steps: int = 1000
    ) -> None:
        r"""
        Args:
            cfg_drop_ratio (float): Dropout ratio for the autoencoder.
            sample_strategy (str): Sampling strategy for timesteps during training.
            num_train_steps (int): Number of training steps for the noise scheduler.
        """
        self.sample_strategy = sample_strategy
        # self.infer_noise_scheduler = FlowMatchEulerDiscreteScheduler(
        #     num_train_timesteps=num_train_steps
        # )
        # self.train_noise_scheduler = copy.deepcopy(self.infer_noise_scheduler)

        self.classifier_free_guidance = cfg_drop_ratio > 0.0
        self.cfg_drop_ratio = cfg_drop_ratio

    def get_input_target_and_timesteps(
        self,
        latent: torch.Tensor,
        training: bool,
    ):
        batch_size = latent.shape[0]
        noise = torch.randn_like(latent)

        if training:
            if self.sample_strategy == 'normal':
                t = compute_density_for_timestep_sampling(
                    weighting_scheme="logit_normal",
                    batch_size=batch_size,
                    logit_mean=0,
                    logit_std=1,
                    mode_scale=None,
                )
            elif self.sample_strategy == 'uniform':
                t = torch.rand(batch_size)
            else:
                raise NotImplementedError(
                    f"{self.sample_strategy} samlping for timesteps is not supported now"
                )
        else:
            t = torch.tensor([0.5] * batch_size)

        t = t.to(latent.device)
        t_expanded = t
        while t_expanded.ndim < latent.ndim:
            t_expanded = t_expanded.unsqueeze(-1)
        noisy_latent = (1.0 - t_expanded) * noise + t_expanded * latent

        target = latent - noise

        return noisy_latent, target, t

    # def retrieve_timesteps(
    #     self,
    #     num_inference_steps: int | None = None,
    #     device: str | torch.device | None = None,
    #     timesteps: list[int] | None = None,
    #     sigmas: list[float] | None = None,
    #     **kwargs,
    # ):
    #     # used in inference, retrieve new timesteps on given inference timesteps
    #     scheduler = self.infer_noise_scheduler

    #     if timesteps is not None and sigmas is not None:
    #         raise ValueError(
    #             "Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values"
    #         )
    #     if timesteps is not None:
    #         accepts_timesteps = "timesteps" in set(
    #             inspect.signature(scheduler.set_timesteps).parameters.keys()
    #         )
    #         if not accepts_timesteps:
    #             raise ValueError(
    #                 f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
    #                 f" timestep schedules. Please check whether you are using the correct scheduler."
    #             )
    #         scheduler.set_timesteps(
    #             timesteps=timesteps, device=device, **kwargs
    #         )
    #         timesteps = scheduler.timesteps
    #         num_inference_steps = len(timesteps)
    #     elif sigmas is not None:
    #         accept_sigmas = "sigmas" in set(
    #             inspect.signature(scheduler.set_timesteps).parameters.keys()
    #         )
    #         if not accept_sigmas:
    #             raise ValueError(
    #                 f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
    #                 f" sigmas schedules. Please check whether you are using the correct scheduler."
    #             )
    #         scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
    #         timesteps = scheduler.timesteps
    #         num_inference_steps = len(timesteps)
    #     else:
    #         scheduler.set_timesteps(
    #             num_inference_steps, device=device, **kwargs
    #         )
    #         timesteps = scheduler.timesteps
    #     return timesteps, num_inference_steps


class MultiConditionFlowMatchingMixin(FlowMatchingMixin):
    def __init__(
        self,
        cfg_drop_config: dict | float,
        sample_strategy: str = "uniform",
        num_train_steps: int = 1000
    ) -> None:
        super().__init__(
            cfg_drop_ratio=0.0,
            sample_strategy=sample_strategy,
            num_train_steps=num_train_steps
        )
        if isinstance(cfg_drop_config, float):
            self.cfg_drop_config = {"all": cfg_drop_config}
        elif isinstance(cfg_drop_config, dict):
            self.cfg_drop_config = cfg_drop_config
        else:
            raise ValueError(f"Invalid cfg_drop_config: {cfg_drop_config}")
        self.classifier_free_guidance = any(
            v > 0.0 for v in self.cfg_drop_config.values()
        )

    def drop_content(
        self, content_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        # FIXME `content_dict` is modified in-place, should be a deep copy, but deep copy may increase memory usage
        for content_key, content in content_dict.items():
            if random() < self.cfg_drop_config[content_key]:
                content[content_key] = torch.zeros_like(content)
        return content_dict


class F5TTSDropTextEmbeddingFlowMatching(
    MultiConditionFlowMatchingMixin, LoadPretrainedBase, CountParamsBase
):
    def __init__(
        self,
        text_encoder: nn.Module,
        backbone: nn.Module,
        tokenizer_path: str,
        tokenizer: str = "char",
        cfg_drop_config: dict | float | None = None,
        sample_strategy: str = "uniform",
        num_train_steps: int = 1000,
        frac_lengths_mask: tuple[float, float] = (0.7, 1.0),
    ) -> None:
        nn.Module.__init__(self)
        if cfg_drop_config is None:
            cfg_drop_config = {"masked_mel": 0.3, "text_embed": 0.2}
        super().__init__(
            cfg_drop_config=cfg_drop_config,
            sample_strategy=sample_strategy,
            num_train_steps=num_train_steps
        )
        self.text_encoder = text_encoder
        self.backbone = backbone
        self.vocab_char_map = get_tokenizer(tokenizer_path, tokenizer)[0]
        self.frac_lengths_mask = frac_lengths_mask

    def drop_content(
        self, content_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        # FIXME `content_dict` is modified in-place, should be a deep copy, but deep copy may increase memory usage
        if random() < self.cfg_drop_config["text_embed"]:
            # NOTE This is different from F5-TTS, F5-TTS set zero to text tokens, so tokens with index 0 are still encoded
            content_dict["text_embed"] = torch.zeros_like(
                content_dict["text_embed"]
            )
            content_dict["masked_mel"] = torch.zeros_like(
                content_dict["masked_mel"]
            )
        else:
            if random() < self.cfg_drop_config["masked_mel"]:
                content_dict["masked_mel"] = torch.zeros_like(
                    content_dict["masked_mel"]
                )
        return content_dict

    def encode_text(
        self, text: torch.Tensor, target_length: int
    ) -> torch.Tensor:
        text_embed = self.text_encoder(text, target_length)
        return text_embed

    def get_backbone_input(
        self, masked_mel: float["b n d"], text_embed: float["b n d"]
    ):
        return torch.cat([masked_mel, text_embed], dim=-1)

    def forward(
        self,
        mel_spec: torch.Tensor,  # ground truth mel spectrogram
        text: list[str],
        mel_spec_lengths: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        batch, seq_len = mel_spec.shape[:2]
        dtype, device = mel_spec.dtype, mel_spec.device

        if isinstance(text, list):
            if exists(self.vocab_char_map):
                text = list_str_to_idx(text, self.vocab_char_map).to(device)
            else:
                text = list_str_to_tensor(text).to(device)
            assert text.shape[0] == batch

        # lens and mask
        if not exists(mel_spec_lengths):
            # if lens not acquired by trainer from collate_fn
            mel_spec_lengths = torch.full((batch, ), seq_len, device=device)
        mask = lens_to_mask(mel_spec_lengths, max_length=seq_len)

        # get a random span to mask out for training conditionally
        frac_lengths = torch.zeros((batch, ), device=device).float().uniform_(
            *self.frac_lengths_mask
        )
        rand_span_mask = mask_from_frac_lengths(mel_spec_lengths, frac_lengths)

        if exists(mask):
            rand_span_mask &= mask

        noisy_mel, target, timesteps = self.get_input_target_and_timesteps(
            mel_spec, training=self.training
        )

        # only predict what is within the random mask span for infilling
        masked_mel = torch.where(
            rand_span_mask[..., None], torch.zeros_like(mel_spec), mel_spec
        )

        text_embedding = self.encode_text(text, mel_spec.shape[1])

        content_dict = {"masked_mel": masked_mel, "text_embed": text_embedding}
        if self.training and self.classifier_free_guidance:
            content_dict = self.drop_content(content_dict)
        time_aligned_content = self.get_backbone_input(**content_dict)

        pred: torch.Tensor = self.backbone(
            x=noisy_mel,
            time=timesteps,
            # context=content,
            x_mask=mask,
            time_aligned_content=time_aligned_content,
            # context_mask=content_mask
        )

        loss = F.mse_loss(pred, target, reduction="none")
        loss = loss[rand_span_mask]
        return loss.mean()

    def iterative_denoise(
        self,
        latent: torch.Tensor,
        timesteps: list[int],
        # noise_scheduler: FlowMatchEulerDiscreteScheduler,
        num_steps: int,
        verbose: bool,
        cfg: bool,
        cfg_scale: float,
        backbone_input: dict,
    ):
        # progress_bar = tqdm(range(num_steps), disable=not verbose)

        # for t0, t1 in zip(timesteps[:-1], timesteps[1:]):
        #     # expand the latent if we are doing classifier free guidance
        #     if cfg:
        #         latent_input = torch.cat([latent, latent])
        #     else:
        #         latent_input = latent

        #     noise_pred: torch.Tensor = self.backbone(
        #         x=latent_input, time=t0, **backbone_input
        #     )

        #     # perform guidance
        #     if cfg:
        #         noise_pred_uncond, noise_pred_content = noise_pred.chunk(2)
        #         noise_pred = noise_pred_uncond + cfg_scale * (
        #             noise_pred_content - noise_pred_uncond
        #         )

        #     dt = t1 - t0
        #     latent = latent + dt * noise_pred

        #     progress_bar.update(1)

        # progress_bar.close()

        # result = latent

        # neural ode

        def fn(t, x):

            if cfg:
                x = torch.cat([x, x])

            noise_pred = self.backbone(
                x=x,
                time=t,
                **backbone_input,
            )
            if cfg:
                pred_uncond, pred_cond = torch.chunk(noise_pred, 2, dim=0)
                noise_pred = pred_uncond + cfg_scale * (
                    pred_cond - pred_uncond
                )

            return noise_pred

        odeint_kwargs = {"method": "euler"}
        trajectory = odeint(fn, latent, timesteps, **odeint_kwargs)
        result = trajectory[-1]

        return result

    @torch.inference_mode()
    def inference(
        self,
        mel_spec: "float['b n d']",
        text: "int['b nt'] | list[str]",
        duration: "int | int['b']",
        latent_shape: Sequence[int],
        # noise_scheduler: FlowMatchEulerDiscreteScheduler,
        mel_spec_lengths: "int['b'] | None" = None,
        num_steps: int = 20,
        use_epss: bool = True,
        sway_sampling_coef: float | None = -1.0,
        guidance_scale: float = 3.0,
        disable_progress: bool = True,
        **kwargs
    ) -> torch.Tensor:
        device = mel_spec.device
        batch, mel_spec_seq_len = mel_spec.shape[0], mel_spec.shape[1]
        classifier_free_guidance = guidance_scale > 1.0

        # text

        if isinstance(text, list):
            if exists(self.vocab_char_map):
                text = list_str_to_idx(text, self.vocab_char_map).to(device)
            else:
                text = list_str_to_tensor(text).to(device)
            assert text.shape[0] == batch

        # duration
        if not exists(mel_spec_lengths):
            mel_spec_lengths = torch.full((batch, ),
                                          mel_spec_seq_len,
                                          device=device)
        mel_spec_mask = lens_to_mask(mel_spec_lengths)
        # if edit_mask is not None:
        #     mel_spec_mask = mel_spec_mask & edit_mask

        if isinstance(duration, int):
            duration = torch.full((batch, ),
                                  duration,
                                  device=device,
                                  dtype=torch.long)

        duration = torch.maximum(
            torch.maximum((text != -1).sum(dim=-1), mel_spec_lengths) + 1,
            duration
        )  # duration at least text/audio prompt length plus one token, so something is generated
        max_duration = duration.amax()

        # padded mel spec (to max duration), padded area to be generated
        mel_spec = F.pad(
            mel_spec, (0, 0, 0, max_duration - mel_spec_seq_len), value=0.0
        )
        mel_spec_mask = F.pad(
            mel_spec_mask, (0, max_duration - mel_spec_mask.shape[1]),
            value=False
        )
        masked_mel = torch.where(
            mel_spec_mask.unsqueeze(-1), mel_spec, torch.zeros_like(mel_spec)
        )

        latent_mask = lens_to_mask(duration)

        # prepare content input to the backbone
        text_embedding = self.encode_text(text, mel_spec.shape[1])
        content_dict = {"masked_mel": masked_mel, "text_embed": text_embedding}

        # --------------------------------------------------------------------
        # prepare unconditional input
        # --------------------------------------------------------------------
        if classifier_free_guidance:
            for key in self.cfg_drop_config:
                uncond_content = torch.zeros_like(content_dict[key])
                content_dict[key] = torch.cat([
                    uncond_content, content_dict[key]
                ])

            latent_mask = torch.cat([
                latent_mask, latent_mask.detach().clone()
            ])

        time_aligned_content = self.get_backbone_input(**content_dict)

        # prepare noise input to the backbone
        latent_shape = tuple(
            max_duration if dim is None else dim for dim in latent_shape
        )
        shape = (batch, *latent_shape)
        mel_spec_generated = randn_tensor(
            shape, generator=None, device=device, dtype=mel_spec.dtype
        )

        if use_epss:  # use Empirically Pruned Step Sampling for low NFE
            t = get_epss_timesteps(
                num_steps, device=device, dtype=mel_spec.dtype
            )
        else:
            t = torch.linspace(
                0, 1, num_steps + 1, device=self.device, dtype=mel_spec.dtype
            )
        if sway_sampling_coef is not None:
            t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)

        # noise_scheduler.set_timesteps(sigmas=t.cpu(), device=device)
        mel_spec_generated = self.iterative_denoise(
            latent=mel_spec_generated,
            timesteps=t,
            # noise_scheduler=noise_scheduler,
            num_steps=num_steps,
            verbose=not disable_progress,
            cfg=classifier_free_guidance,
            cfg_scale=guidance_scale,
            backbone_input={
                "x_mask": latent_mask,
                "time_aligned_content": time_aligned_content,
            }
        )

        mel_spec_generated = torch.where(
            mel_spec_mask.unsqueeze(-1), mel_spec, mel_spec_generated
        )

        return mel_spec_generated


class F5TTSDropTextTokenFlowMatching(F5TTSDropTextEmbeddingFlowMatching):
    def encode_text(
        self,
        text: torch.Tensor,
        target_length: int,
        drop_text: bool = False
    ) -> torch.Tensor:
        text_embed = self.text_encoder(
            text, target_length, drop_text=drop_text
        )
        return text_embed

    def forward(
        self,
        mel_spec: torch.Tensor,
        text: list[str],
        mel_spec_lengths: torch.Tensor | None = None,
        **kwargs
    ) -> torch.Tensor:
        batch, seq_len = mel_spec.shape[:2]
        dtype, device = mel_spec.dtype, mel_spec.device

        if isinstance(text, list):
            if exists(self.vocab_char_map):
                text = list_str_to_idx(text, self.vocab_char_map).to(device)
            else:
                text = list_str_to_tensor(text).to(device)
            assert text.shape[0] == batch

        # lens and mask
        if not exists(mel_spec_lengths):
            # if lens not acquired by trainer from collate_fn
            mel_spec_lengths = torch.full((batch, ), seq_len, device=device)
        mask = lens_to_mask(mel_spec_lengths, max_length=seq_len)

        # get a random span to mask out for training conditionally
        frac_lengths = torch.zeros((batch, ), device=device).float().uniform_(
            *self.frac_lengths_mask
        )
        rand_span_mask = mask_from_frac_lengths(mel_spec_lengths, frac_lengths)

        if exists(mask):
            rand_span_mask &= mask

        noisy_mel, target, timesteps = self.get_input_target_and_timesteps(
            mel_spec, training=self.training
        )

        # only predict what is within the random mask span for infilling
        masked_mel = torch.where(
            rand_span_mask[..., None], torch.zeros_like(mel_spec), mel_spec
        )

        if self.training and self.classifier_free_guidance:
            drop_text = random() < self.cfg_drop_config["text_embed"]
            if drop_text:
                drop_audio = True
            else:
                drop_audio = random() < self.cfg_drop_config["masked_mel"]
        else:
            drop_audio, drop_text = False, False

        text_embedding = self.encode_text(
            text, mel_spec.shape[1], drop_text=drop_text
        )
        if drop_audio:
            masked_mel = torch.zeros_like(masked_mel)
        content_dict = {"masked_mel": masked_mel, "text_embed": text_embedding}

        time_aligned_content = self.get_backbone_input(**content_dict)

        pred: torch.Tensor = self.backbone(
            x=noisy_mel,
            time=timesteps,
            # context=content,
            x_mask=mask,
            time_aligned_content=time_aligned_content,
            # context_mask=content_mask
        )

        loss = F.mse_loss(pred, target, reduction="none")
        loss = loss[rand_span_mask]
        return loss.mean()

    @torch.inference_mode()
    def inference(
        self,
        mel_spec: "float['b n d']",
        text: "int['b nt'] | list[str]",
        duration: "int | int['b']",
        latent_shape: Sequence[int],
        # noise_scheduler: FlowMatchEulerDiscreteScheduler,
        mel_spec_lengths: "int['b'] | None" = None,
        num_steps: int = 20,
        use_epss: bool = True,
        sway_sampling_coef: float | None = -1.0,
        guidance_scale: float = 3.0,
        disable_progress: bool = True,
        **kwargs
    ) -> torch.Tensor:
        device = mel_spec.device
        batch, mel_spec_seq_len = mel_spec.shape[0], mel_spec.shape[1]
        classifier_free_guidance = guidance_scale > 1.0

        # text

        if isinstance(text, list):
            if exists(self.vocab_char_map):
                text = list_str_to_idx(text, self.vocab_char_map).to(device)
            else:
                text = list_str_to_tensor(text).to(device)
            assert text.shape[0] == batch

        # duration
        if not exists(mel_spec_lengths):
            mel_spec_lengths = torch.full((batch, ),
                                          mel_spec_seq_len,
                                          device=device)
        mel_spec_mask = lens_to_mask(mel_spec_lengths)
        # if edit_mask is not None:
        #     mel_spec_mask = mel_spec_mask & edit_mask

        if isinstance(duration, int):
            duration = torch.full((batch, ),
                                  duration,
                                  device=device,
                                  dtype=torch.long)

        duration = torch.maximum(
            torch.maximum((text != -1).sum(dim=-1), mel_spec_lengths) + 1,
            duration
        )  # duration at least text/audio prompt length plus one token, so something is generated
        max_duration = duration.amax()

        # padded mel spec (to max duration), padded area to be generated
        mel_spec = F.pad(
            mel_spec, (0, 0, 0, max_duration - mel_spec_seq_len), value=0.0
        )
        mel_spec_mask = F.pad(
            mel_spec_mask, (0, max_duration - mel_spec_mask.shape[1]),
            value=False
        )
        masked_mel = torch.where(
            mel_spec_mask.unsqueeze(-1), mel_spec, torch.zeros_like(mel_spec)
        )

        latent_mask = lens_to_mask(duration)

        # prepare content input to the backbone
        text_embedding = self.encode_text(text, mel_spec.shape[1])
        uncond_text_embedding = self.encode_text(
            text, mel_spec.shape[1], drop_text=True
        )
        content_dict = {"masked_mel": masked_mel, "text_embed": text_embedding}

        # --------------------------------------------------------------------
        # prepare unconditional input
        # --------------------------------------------------------------------
        if classifier_free_guidance:
            for key in self.cfg_drop_config:
                if key != "text_embed":
                    uncond_content = torch.zeros_like(content_dict[key])
                else:
                    uncond_content = uncond_text_embedding
                content_dict[key] = torch.cat([
                    uncond_content, content_dict[key]
                ])

            latent_mask = torch.cat([
                latent_mask, latent_mask.detach().clone()
            ])

        time_aligned_content = self.get_backbone_input(**content_dict)

        # prepare noise input to the backbone
        latent_shape = tuple(
            max_duration if dim is None else dim for dim in latent_shape
        )
        shape = (batch, *latent_shape)
        mel_spec_generated = randn_tensor(
            shape, generator=None, device=device, dtype=mel_spec.dtype
        )

        if use_epss:  # use Empirically Pruned Step Sampling for low NFE
            t = get_epss_timesteps(
                num_steps, device=device, dtype=mel_spec.dtype
            )
        else:
            t = torch.linspace(
                0, 1, num_steps + 1, device=self.device, dtype=mel_spec.dtype
            )
        if sway_sampling_coef is not None:
            t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)

        # noise_scheduler.set_timesteps(sigmas=t.cpu(), device=device)
        mel_spec_generated = self.iterative_denoise(
            latent=mel_spec_generated,
            timesteps=t,
            # noise_scheduler=noise_scheduler,
            num_steps=num_steps,
            verbose=not disable_progress,
            cfg=classifier_free_guidance,
            cfg_scale=guidance_scale,
            backbone_input={
                "x_mask": latent_mask,
                "time_aligned_content": time_aligned_content,
            }
        )

        mel_spec_generated = torch.where(
            mel_spec_mask.unsqueeze(-1), mel_spec, mel_spec_generated
        )

        return mel_spec_generated
