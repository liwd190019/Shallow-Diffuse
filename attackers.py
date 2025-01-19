import torch
from torchvision import transforms
import os

from PIL import Image, ImageFilter
import random
import numpy as np
from typing import Any, Mapping
from attdiffuse import ReSDPipeline
from compressai.zoo import bmshj2018_factorized, bmshj2018_hyperprior, mbt2018_mean, mbt2018, cheng2020_anchor


# Initialize global variables for VAE and Diffusion attackers
vae_attacker1 = None
vae_attacker2 = None
diff_attacker = None

def initialize_attackers(args, device):
    global vae_attacker1, vae_attacker2, diff_attacker
    # Initialize VAE-based attacker
    if args.vae_attack_model_name1 is not None and args.vae_attack_model_name2 is not None:
        vae_attacker1 = VAEWMAttacker(args.vae_attack_model_name1, quality=3, metric='mse', device=device)
        vae_attacker2 = VAEWMAttacker(args.vae_attack_model_name2, quality=3, metric='mse', device=device)

    # Initialize Diffusion-based attacker
    att_pipe = ReSDPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16, revision="fp16")
    att_pipe.set_progress_bar_config(disable=True)
    att_pipe.to(device)
    diff_attacker = DiffWMAttacker(att_pipe, batch_size=1, noise_step=60, captions={})


def image_distortion_none(imgs: list, seed, args):
    return

def image_distortion_rotation(imgs: list, seed, args):
    rotated_imgs = []
    rotation_transform = transforms.RandomRotation((args.r_degree, args.r_degree))
    for img in imgs:
        img.rotation = rotation_transform(img.none)


def image_distortion_jpeg(imgs: list, seed: int, args):
    distorted_imgs = []
    for i, img in enumerate(imgs):
        temp_filename = f"scratch/tmp_{args.jpeg_ratio}_{args.run_name}.jpg"
        img.none.save(temp_filename, quality=args.jpeg_ratio)
        img.jpeg = Image.open(temp_filename)

def image_distortion_crop(imgs: list, seed: int, args):
    distorted_imgs = []
    for img in imgs:
        set_random_seed(seed)
        crop_transform = transforms.RandomResizedCrop(
            img.none.size, scale=(args.crop_scale, args.crop_scale), ratio=(args.crop_ratio, args.crop_ratio)
        )
        img.crop = crop_transform(img.none)


def image_distortion_gaussianblur(imgs: list, seed: int, args):
    blurred_imgs = []
    for img in imgs:
        img.gaussianblur = img.none.filter(ImageFilter.GaussianBlur(radius=args.gaussian_blur_r))


def image_distortion_gaussianstd(imgs: list, seed: int, args):
    random.seed(seed)  # Set the random seed for reproducibility
    img_shape = np.array(imgs[0]).shape
    g_noise = np.random.normal(0, args.gaussian_std, img_shape) * 255
    g_noise = g_noise.astype(np.uint8)
    distorted_imgs = []
    for img in imgs:
        img.gaussianstd = Image.fromarray(np.clip(np.array(img.none) + g_noise, 0, 255))

def image_distortion_medianblur(imgs: list, seed: int, args):
    random.seed(seed)
    for img in imgs:
        img.medianblur = img.none.filter(ImageFilter.MedianFilter(args.median_blur))

def image_distortion_colorjitter(imgs: list, seed: int, args):
    random.seed(seed)
    distorted_imgs = []
    jitter_transform = transforms.ColorJitter(brightness=args.brightness_factor)
    for img in imgs:
        img.colorjitter = jitter_transform(img.none)

def image_distortion_randomdrop(imgs: list, seed: int, args):
    random.seed(seed)  # Set the random seed for reproducibility
    distorted_imgs = []
    random_drop_ratio = 0.4
    for img in imgs:
        img_array = np.array(img.none)
        height, width, c = img_array.shape
        new_width = int(width * random_drop_ratio)
        new_height = int(height * random_drop_ratio)
        start_x = (width - new_width) // 2
        end_x = start_x + new_width
        start_y = (height - new_height) // 2
        end_y = start_y + new_height
        img_array[start_y:end_y, start_x:end_x] = np.zeros_like(
            img_array[start_y:end_y, start_x:end_y]
        )
        img.randomdrop = Image.fromarray(img_array)

def image_distortion_diffpure(imgs, seed, current_prompt_embeddings, null_embeddings, pipe, args):
    for i, img in enumerate(imgs):
        x0_gt_tensor = transform_img(img.none).unsqueeze(0).to(null_embeddings.dtype).to(null_embeddings.device)
        x0_latent = pipe.get_image_latents(x0_gt_tensor, sample=False)
        x0_latent_purify = pipe.diffusion_purification(
            num_inference_steps = args.num_inference_steps,
            purify_timestep = int(0.3 * args.num_inference_steps),
            latents = x0_latent,
            text_embeddings = current_prompt_embeddings,
            text_embeddings_null = null_embeddings,
            guidance_scale = args.guidance_scale,
        )
        x0_img_purify = pipe.numpy_to_pil(pipe.decode_latents(x0_latent_purify))[0]
        img.diffpure = x0_img_purify

def image_distortion_saltandpepper(imgs: list, seed: int, args):
    random.seed(seed)  # Set the random seed for reproducibility
    np.random.seed(seed)  # Set NumPy's random seed for reproducibility
    distorted_imgs = []
    for img in imgs:
        img_array = np.array(img.none)
        c, h, w = img_array.shape
        sp_prob = 0.05
        prob_zero = sp_prob / 2  # Probability of pepper (black) noise
        prob_one = 1 - prob_zero      # Probability of not applying noise
        rdn = np.random.rand(c, h, w)
        img_array = np.where(rdn > prob_one, np.zeros_like(img_array), img_array)
        img_array = np.where(rdn < prob_zero, np.ones_like(img_array) * 255, img_array)
        img.saltandpepper = Image.fromarray(img_array)

def image_distortion_resizerestore(imgs: list, seed: int, args):
    random.seed(seed)  # Set the random seed for reproducibility
    distorted_imgs = []
    for img in imgs:
        original_size = img.none.size
        new_size = (int(original_size[0] * 0.25), int(original_size[1] * 0.25))
        resized_img = img.none.resize(new_size, Image.LANCZOS)
        restored_img = resized_img.resize(original_size, Image.LANCZOS)
        img.resizerestore = restored_img

def image_distortion_vae1(imgs: list, seed, args):
    """
    VAE-based Compression Attack (using bmshj2018 or mbt2018 models)
    Args:
        imgs (list): List of images to attack.
        args: Arguments containing attack model name and run information.
    """
    temp_paths = []
    output_paths = []

    for i, img in enumerate(imgs):
        tmp_path = f"tmp_img_{i}_{args.run_name}.jpg"
        output_path = f"tmp_vae_{args.vae_attack_model_name1}_{args.run_name}_{i}.jpg"
        img.none.save(tmp_path)  # Save the original image
        temp_paths.append(tmp_path)
        output_paths.append(output_path)

    vae_attacker1.attack(temp_paths, output_paths, multi=False)

    for i, img in enumerate(imgs):
        img.vaebmshj = Image.open(output_paths[i])

    # Clean up temporary files
    for path in temp_paths + output_paths:
        if os.path.exists(path):
            os.remove(path)

def image_distortion_vae2(imgs: list, seed, args):
    """
    VAE-based Compression Attack (using bmshj2018 or mbt2018 models)
    Args:
        imgs (list): List of images to attack.
        args: Arguments containing attack model name and run information.
    """
    temp_paths = []
    output_paths = []

    for i, img in enumerate(imgs):
        tmp_path = f"tmp_img_{i}_{args.run_name}.jpg"
        output_path = f"tmp_vae_{args.vae_attack_model_name2}_{args.run_name}_{i}.jpg"
        img.none.save(tmp_path)  # Save the original image
        temp_paths.append(tmp_path)
        output_paths.append(output_path)
    
    vae_attacker2.attack(temp_paths, output_paths, multi=False)

    for i, img in enumerate(imgs):
        img.vaecheng = Image.open(output_paths[i])

    # Clean up temporary files
    for path in temp_paths + output_paths:
        if os.path.exists(path):
            os.remove(path)

def image_distortion_diff(imgs: list, seed, args):
    """
    Diffusion-based Regeneration Attack (e.g., Zhao23)
    Args:
        imgs (list): List of images to attack.
        args: Arguments containing attack configuration.
    """
    temp_paths = []
    output_paths = []

    for i, img in enumerate(imgs):
        tmp_path = f"tmp_img_{i}_{args.run_name}.jpg"
        output_path = f"tmp_diff_{args.run_name}_{i}.jpg"
        img.none.save(tmp_path)  # Save the original image
        temp_paths.append(tmp_path)
        output_paths.append(output_path)
    
    diff_attacker.attack(temp_paths, output_paths, return_latents=False, multi=False)

    # Reload modified images and store them in the `diff` attribute of each img instance
    for i, img in enumerate(imgs):
        img.diff = Image.open(output_paths[i])

    # Clean up temporary files
    for path in temp_paths + output_paths:
        if os.path.exists(path):
            os.remove(path)

class WMAttacker:
    def attack(self, imgs_path, out_path):
        raise NotImplementedError

class VAEWMAttacker(WMAttacker):
    def __init__(self, model_name, quality=1, metric='mse', device='cpu'):
        if model_name == 'bmshj2018_factorized':
            self.model = bmshj2018_factorized(quality=quality, pretrained=True).eval().to(device)
        elif model_name == 'bmshj2018_hyperprior':
            self.model = bmshj2018_hyperprior(quality=quality, pretrained=True).eval().to(device)
        elif model_name == 'mbt2018_mean':
            self.model = mbt2018_mean(quality=quality, pretrained=True).eval().to(device)
        elif model_name == 'mbt2018':
            self.model = mbt2018(quality=quality, pretrained=True).eval().to(device)
        elif model_name == 'cheng2020_anchor':
            self.model = cheng2020_anchor(quality=quality, pretrained=True).eval().to(device)
        else:
            raise ValueError('model name not supported')
        self.device = device

    def attack(self, image_paths, out_paths, multi=False):
        for (img_path, out_path) in zip(image_paths, out_paths):
            if os.path.exists(out_path) and not multi:
                continue
            
            img = Image.open(img_path).convert('RGB')
            img = img.resize((512, 512))
            img = transforms.ToTensor()(img).unsqueeze(0).to(self.device)
            out = self.model(img)
            out['x_hat'].clamp_(0, 1)
            rec = transforms.ToPILImage()(out['x_hat'].squeeze().cpu())
            rec.save(out_path)

class DiffWMAttacker(WMAttacker):
    def __init__(self, pipe, batch_size=20, noise_step=60, captions={}):
        self.pipe = pipe
        self.BATCH_SIZE = batch_size
        self.device = pipe.device
        self.noise_step = noise_step
        self.captions = captions

    def attack(self, image_paths, out_paths, return_latents=False, return_dist=False, multi=False):
        with torch.no_grad():
            generator = torch.Generator(self.device).manual_seed(1024)
            latents_buf = []
            prompts_buf = []
            outs_buf = []
            timestep = torch.tensor([self.noise_step], dtype=torch.long, device=self.device)
            ret_latents = []

            def batched_attack(latents_buf, prompts_buf, outs_buf):
                latents = torch.cat(latents_buf, dim=0)
                images = self.pipe(prompts_buf,
                                   head_start_latents=latents,
                                   head_start_step=50 - max(self.noise_step // 20, 1),
                                   guidance_scale=7.5,
                                   generator=generator, )
                images = images[0]
                for img, out in zip(images, outs_buf):
                    img.save(out)

            if len(self.captions) != 0:
                prompts = []
                for img_path in image_paths:
                    img_name = os.path.basename(img_path)
                    if img_name[:-4] in self.captions:
                        prompts.append(self.captions[img_name[:-4]])
                    else:
                        prompts.append("")
            else:
                prompts = [""] * len(image_paths)

            for (img_path, out_path), prompt in zip(zip(image_paths, out_paths), prompts):
                if os.path.exists(out_path) and not multi:
                    continue
                
                img = Image.open(img_path)
                img = np.asarray(img) / 255
                img = (img - 0.5) * 2
                img = torch.tensor(img, dtype=torch.float16, device=self.device).permute(2, 0, 1).unsqueeze(0)
                latents = self.pipe.vae.encode(img).latent_dist
                latents = latents.sample(generator) * self.pipe.vae.config.scaling_factor
                noise = torch.randn([1, 4, img.shape[-2] // 8, img.shape[-1] // 8], device=self.device)
                if return_dist:
                    return self.pipe.scheduler.add_noise(latents, noise, timestep, return_dist=True)
                latents = self.pipe.scheduler.add_noise(latents, noise, timestep).type(torch.half)
                latents_buf.append(latents)
                outs_buf.append(out_path)
                prompts_buf.append(prompt)
                if len(latents_buf) == self.BATCH_SIZE:
                    batched_attack(latents_buf, prompts_buf, outs_buf)
                    latents_buf = []
                    prompts_buf = []
                    outs_buf = []
                if return_latents:
                    ret_latents.append(latents.cpu())

            if len(latents_buf) != 0:
                batched_attack(latents_buf, prompts_buf, outs_buf)
            if return_latents:
                return ret_latents
