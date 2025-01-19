from functools import partial
from typing import Callable, List, Optional, Union, Tuple

import torch
import copy
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

from diffusers.models import AutoencoderKL, UNet2DConditionModel
# from diffusers import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion.safety_checker import \
    StableDiffusionSafetyChecker
from diffusers.schedulers import DDIMScheduler,PNDMScheduler, LMSDiscreteScheduler

from modified_stable_diffusion import ModifiedStableDiffusionPipeline

from torchvision.transforms import ToPILImage
import os

from PIL import Image


def backward_ddim(x_t, alpha_t, alpha_tm1, eps_xt):
    """ from noise to image"""
    return (
        alpha_tm1**0.5
        * (
            (alpha_t**-0.5 - alpha_tm1**-0.5) * x_t
            + ((1 / alpha_tm1 - 1) ** 0.5 - (1 / alpha_t - 1) ** 0.5) * eps_xt
        )
        + x_t
    )

def forward_ddim(x_t, alpha_t, alpha_tp1, eps_xt):
    """ from image to noise, it's the same as backward_ddim"""
    return backward_ddim(x_t, alpha_t, alpha_tp1, eps_xt)

def latents_to_image(latents):
    # Assuming the latents are in the correct range and shape, adjust as necessary
    latents = latents.squeeze(0).cpu()
    latents = (latents - latents.min()) / (latents.max() - latents.min())
    to_pil = ToPILImage()
    return to_pil(latents)

@staticmethod
def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]
    return pil_images


class InversableStableDiffusionPipeline(ModifiedStableDiffusionPipeline):
    def __init__(self,
        vae,
        text_encoder,
        tokenizer,
        unet,
        scheduler,
        safety_checker,
        feature_extractor,
        requires_safety_checker: bool = True,
    ):
        super(InversableStableDiffusionPipeline, self).__init__(vae,
                text_encoder,
                tokenizer,
                unet,
                scheduler,
                safety_checker,
                feature_extractor,
                requires_safety_checker)

        self.forward_diffusion = partial(self.backward_diffusion, reverse_process=True)
    
    def get_random_latents(self, latents=None, height=512, width=512, generator=None):
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        batch_size = 1
        device = self._execution_device

        num_channels_latents = self.unet.in_channels

        latents = self.prepare_latents(
            batch_size,
            num_channels_latents,
            height,
            width,
            self.text_encoder.dtype,
            device,
            generator,
            latents,
        )

        return latents

    @torch.inference_mode()
    def get_text_embedding(self, prompt):
        text_input_ids = self.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids
        text_embeddings = self.text_encoder(text_input_ids.to(self.device))[0]
        return text_embeddings
    
    @torch.inference_mode()
    def get_image_latents(self, image, sample=True, rng_generator=None):
        encoding_dist = self.vae.encode(image).latent_dist
        if sample:
            encoding = encoding_dist.sample(generator=rng_generator)
        else:
            encoding = encoding_dist.mode()
        latents = encoding * 0.18215
        return latents



    @torch.inference_mode()
    def backward_diffusion(
        self,
        use_old_emb_i=25,
        text_embeddings=None,
        text_embeddings_null=None,
        latents: Optional[torch.FloatTensor] = None,
        num_inference_steps: int = 100,
        guidance_scale: float = 7.5,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        reverse_process: bool = False,
        start_timestep = 0,
        end_timestep = 999,
        save_path = './output',
        **kwargs,
    ):
        """ Generate image from text prompt and latents
        """
        do_classifier_free_guidance = guidance_scale > 1.0
        self.scheduler.set_timesteps(num_inference_steps)
        # Some schedulers like PNDM have timesteps as arrays
        # It's more optimized to move all timesteps to correct device beforehand
        timesteps_tensor = self.scheduler.timesteps.to(self.device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma

        for i, t in enumerate(self.progress_bar(timesteps_tensor if not reverse_process else reversed(timesteps_tensor))):
            if i < start_timestep:
                continue
            elif i == end_timestep:
                return latents
            elif i == start_timestep:
                pass
            else:
                pass

            # expand the latents if we are doing classifier free guidance
            latent_model_input = (
                torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            )

            new_text_embeddings = copy.deepcopy(text_embeddings)
            if do_classifier_free_guidance:
                new_text_embeddings = torch.cat([text_embeddings_null, new_text_embeddings])
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)


            prev_timestep = (
                t - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
            )
            t_prev = t - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps

            noise_pred = self.unet(
                     latent_model_input, t, encoder_hidden_states=new_text_embeddings
                 ).sample
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # ddim
            alpha_prod_t = self.scheduler.alphas_cumprod[t]
            alpha_prod_t_prev = (
                self.scheduler.alphas_cumprod[prev_timestep]
                if prev_timestep >= 0
                else self.scheduler.final_alpha_cumprod
            )
            if reverse_process:
                alpha_prod_t, alpha_prod_t_prev = alpha_prod_t_prev, alpha_prod_t
            
            # apply ddim backward step: xt->xtm1
            latents = backward_ddim(
                x_t=latents,
                alpha_t=alpha_prod_t,
                alpha_tm1=alpha_prod_t_prev,
                eps_xt=noise_pred,
            )

        return latents

    @torch.inference_mode()
    def diffusion_purification(
        self,
        latents,
        purify_timestep,
        num_inference_steps,
        text_embeddings,
        text_embeddings_null,
        guidance_scale,
        **kwargs,
    ):
        self.scheduler.set_timesteps(num_inference_steps)
        # Some schedulers like PNDM have timesteps as arrays
        # It's more optimized to move all timesteps to correct device beforehand
        timesteps_tensor = torch.flip(self.scheduler.timesteps.to(self.device), dims=[0])
        alpha_t = self.scheduler.alphas_cumprod[timesteps_tensor[purify_timestep]].sqrt()
        beta_t = torch.sqrt(1 - alpha_t**2)
        latents_noised = latents * alpha_t + torch.randn_like(latents) * beta_t
        latents_purify = self.backward_diffusion(
                        latents = latents_noised,
                        text_embeddings = text_embeddings,
                        text_embeddings_null = text_embeddings_null,
                        guidance_scale = guidance_scale,
                        num_inference_steps = num_inference_steps,
                        reverse_process = False,
                        start_timestep = num_inference_steps - purify_timestep - 1,
                        end_timestep = -1,
                    )
        return latents_purify

    
    @torch.inference_mode()
    def decode_image(self, latents: torch.FloatTensor, **kwargs):
        scaled_latents = 1 / 0.18215 * latents
        image = [
            self.vae.decode(scaled_latents[i : i + 1]).sample for i in range(len(latents))
            # self.vae.decode(scaled_latents[i : i + 1]).mode() for i in range(len(latents))
        ]
        image = torch.cat(image, dim=0)
        return image
    
    # from pipeline_stable_diffusion.py
    # home/wdli/scratch/miniconda3/lib/python3.10/site-package/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py -> line 339
    def decode_latents(self, latents):
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        image = image.cpu().permute(0, 2, 3, 1).float().detach().numpy()
        return image

    @torch.inference_mode()
    def torch_to_numpy(self, image):
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        return image
