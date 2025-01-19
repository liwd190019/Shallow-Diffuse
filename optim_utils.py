import torch
from torchvision import transforms
from datasets import load_dataset
import os

from PIL import Image, ImageFilter
import random
import numpy as np
import copy
from typing import Any, Mapping
import json
import scipy
from sklearn import metrics
from statistics import mean, stdev

import torch.nn.functional as F

class one_image:
    def __init__(self, clear_img, label):
        self.none = clear_img
        self.label = label
        self.jpeg = None
        self.gaussianblur = None
        self.gaussianstd = None
        self.colorjitter = None
        self.randomdrop = None
        self.saltandpepper = None
        self.resizerestore = None
        self.vaebmshj = None
        self.vaecheng = None
        self.diff = None
        self.diffpure = None
        self.medianblur = None
        self.reversed_latent_zt_none = None
        self.reversed_latent_zt_jpeg = None
        self.reversed_latent_zt_gaussainblur = None
        self.reversed_latent_zt_gaussianstd = None
        self.reversed_latent_zt_colorjitter = None
        self.reversed_latent_zt_randomdrop = None
        self.reversed_latent_zt_saltandpepper = None
        self.reversed_latent_zt_resizerestore = None
        self.reversed_latent_zt_vaebmshj = None
        self.reversed_latent_zt_vaecheng = None
        self.reversed_latent_zt_diff = None
        self.reversed_latent_zt_medianblur = None
        self.reversed_latent_zt_diffpure = None

def set_random_seed(seed=0):
    torch.manual_seed(seed + 0)
    torch.cuda.manual_seed(seed + 1)
    torch.cuda.manual_seed_all(seed + 2)
    np.random.seed(seed + 3)
    torch.cuda.manual_seed_all(seed + 4)
    random.seed(seed + 5)


def transform_img(image, target_size=512):
    tform = transforms.Compose(
        [
            transforms.Resize(target_size),     # resize the image
            transforms.CenterCrop(target_size), # center cropping the image
            transforms.ToTensor(),              # converting a PIL image or numpy array to tensor
        ]
    )
    image = tform(image)
    return 2.0 * image - 1.0

def measure_similarity(images, prompt, model, clip_preprocess, tokenizer, device):
    with torch.no_grad():
        img_batch = [clip_preprocess(i).unsqueeze(0) for i in images]
        img_batch = torch.concatenate(img_batch).to(device)
        image_features = model.encode_image(img_batch)

        text = tokenizer([prompt]).to(device)
        text_features = model.encode_text(text)
        
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        return (image_features @ text_features.T).mean(-1)

def get_dataset(args):
    if 'laion' in args.dataset:
        dataset = load_dataset(args.dataset)['train']
        prompt_key = 'TEXT'
    elif 'coco' in args.dataset:
        with open('fid_outputs/coco/meta_data.json') as f:
            dataset = json.load(f)
            dataset = dataset['annotations']
            prompt_key = 'caption'
    else:
        dataset = load_dataset(args.dataset)['test']
        prompt_key = 'Prompt'
    return dataset, prompt_key

def get_dataset2(args):
    r"""directly load images"""
    dataset = os.listdir(args.dataset)
    return dataset

def print_tensor_statistics(tensor, tensor_name):
    max_value = tensor.max().item()
    min_value = tensor.min().item()
    mean_value = tensor.mean().item()
    var_value = tensor.var().item()
    print(f'{tensor_name}: max = {max_value}, min = {min_value}, mean = {mean_value}, var = {var_value}')

def circle_mask(size=64, r=10, x_offset=0, y_offset=0):
    # reference: https://stackoverflow.com/questions/69687798/generating-a-soft-circluar-mask-using-numpy-python-3
    x0 = y0 = size // 2
    x0 += x_offset
    y0 += y_offset
    y, x = np.ogrid[:size, :size]
    y = y[::-1]
    if r >= 0:
        return ((x - x0)**2 + (y-y0)**2)<= r**2
    else:
        return ((x - x0)**2 + (y-y0)**2) <= -1

def get_watermarking_mask(init_latents_w=None, w_mask_shape=None, w_radius=None, w_radius2=None, w_channel=None, device=None):
    watermarking_mask = torch.zeros(init_latents_w.shape, dtype=torch.bool).to(device)
    if w_mask_shape == 'circle':
        np_mask = circle_mask(init_latents_w.shape[-1], r=w_radius)
        torch_mask = torch.tensor(np_mask).to(device)
        if w_channel == -1: # all channels
            watermarking_mask[:, :] = torch_mask
        else:
            watermarking_mask[:, w_channel] = torch_mask
    elif w_mask_shape == 'ring':
        outer_radius = w_radius
        inner_radius = w_radius2
        outer_mask = circle_mask(init_latents_w.shape[-1], r=outer_radius)
        inner_mask = circle_mask(init_latents_w.shape[-1], r=inner_radius)
        torch_outer_mask = torch.tensor(outer_mask).to(device)
        torch_inner_mask = torch.tensor(inner_mask).to(device)
        ring_mask = torch_outer_mask & ~torch_inner_mask
        if w_channel == -1:  # all channels
            watermarking_mask[:, :] = ring_mask
        else:
            watermarking_mask[:, w_channel] = ring_mask
    elif w_mask_shape == 'square':
        anchor_p = init_latents_w.shape[-1] // 2
        if w_channel == -1:
            # all channels
            watermarking_mask[:, :, anchor_p-w_radius:anchor_p+w_radius, anchor_p-w_radius:anchor_p+w_radius] = True
        else:
            watermarking_mask[:, w_channel, anchor_p-w_radius:anchor_p+w_radius, anchor_p-w_radius:anchor_p+w_radius] = True
    elif w_mask_shape == 'whole':
        if w_channel == -1:
            # all channels
            watermarking_mask[:, :] = True
        else:
            watermarking_mask[:, w_channel] = True
    elif w_mask_shape == 'outercircle':
        np_mask = circle_mask(init_latents_w.shape[-1], r=w_radius)
        torch_mask = torch.tensor(~np_mask).to(device)  # Invert the circle mask to get the outer circle
        if w_channel == -1: # all channels
            watermarking_mask[:, :] = torch_mask
        else:
            watermarking_mask[:, w_channel] = torch_mask
    else:
        raise NotImplementedError(f'w_mask_shape: {w_mask_shape}')
    return watermarking_mask

def get_rewatermarking_mask(init_latents_w, divisions, index, w_radius, w_channel, device):
    if divisions < 1 or index < 0 or index >= divisions:
        raise ValueError(f"Invalid divisions ({divisions}) or index ({index}). Ensure 0 <= index < divisions.")
    watermarking_mask = torch.zeros(init_latents_w.shape, dtype=torch.bool, device=device)

    # Generate the angle map
    size = init_latents_w.shape[-1]
    angle_map = np.arctan2(
        np.arange(size) - size // 2,
        np.arange(size)[:, None] - size // 2
    )
    angle_map = np.rad2deg(angle_map) % 360  # Convert angles to degrees [0, 360)
    angle_per_division = 360 / divisions
    start_angle = index * angle_per_division
    end_angle = (index + 1) * angle_per_division

    # Create the angular mask
    angular_mask = (angle_map >= start_angle) & (angle_map < end_angle)

    # Create the circular mask
    np_mask = circle_mask(size, r=w_radius)
    combined_mask = np_mask & angular_mask

    torch_mask = torch.tensor(combined_mask, dtype=torch.bool, device=device)
    if w_channel == -1:  # Apply to all channels
        watermarking_mask[:, :] = torch_mask
    else:  # Apply to a specific channel
        watermarking_mask[:, w_channel] = torch_mask

    return watermarking_mask

def get_watermarking_pattern(pipe, args, device, shape=None, this_seed=11):
    set_random_seed(args.w_seed)
    if shape is not None:
        gt_init = torch.randn(*shape, device=device)
    else:
        gt_init = pipe.get_random_latents()
    if 'seed_ring' in args.w_pattern:
        gt_patch = gt_init
        gt_patch_tmp = copy.deepcopy(gt_patch)
        for i in range(args.w_radius, 0, -1):
            tmp_mask = circle_mask(gt_init.shape[-1], r=i)
            tmp_mask = torch.tensor(tmp_mask).to(device)
            for j in range(gt_patch.shape[1]):
                gt_patch[:, j, tmp_mask] = gt_patch_tmp[0, j, 0, i].item()
    elif 'seed_zero' in args.w_pattern:
        gt_patch = gt_init * 0
    elif 'seed_rand' in args.w_pattern:
        gt_patch = gt_init
    elif 'complex2_rand' in args.w_pattern:
        gt_patch = torch.fft.fft2(gt_init)
        gt_patch[:] = gt_patch[0]
    elif 'complex2_zero' in args.w_pattern:
        gt_patch = torch.fft.fft2(gt_init) * 0
    elif 'complex2_ring' in args.w_pattern:
        gt_patch = torch.fft.fft2(gt_init)
        gt_patch_tmp = copy.deepcopy(gt_patch)
        for i in range(args.w_radius, 0, -1):
            tmp_mask = circle_mask(gt_init.shape[-1], r=i)
            tmp_mask = torch.tensor(tmp_mask).to(device)
            for j in range(gt_patch.shape[1]):
                gt_patch[:, j, tmp_mask] = gt_patch_tmp[0, j, 0, i].item()
    elif 'complex_zero' in args.w_pattern:
        gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2)) * 0
    elif 'complex_ring' in args.w_pattern:
        gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2))
        gt_patch_tmp = copy.deepcopy(gt_patch)
        for i in range(args.w_radius, 0, -1):
            tmp_mask = circle_mask(gt_init.shape[-1], r=i)
            tmp_mask = torch.tensor(tmp_mask).to(device)
            for j in range(gt_patch.shape[1]):
                gt_patch[:, j, tmp_mask] = gt_patch_tmp[0, j, 0, i].item()
    elif 'complex_rand' in args.w_pattern:
        gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2))
        gt_patch[:] = gt_patch[0]
    return gt_patch

def inject_watermark(init_latents_w, watermarking_mask, gt_patch, w_injection):
    if 'complex' == w_injection:
        init_latents_w_fft = torch.fft.fftshift(torch.fft.fft2(init_latents_w), dim=(-1, -2))
        init_latents_w_fft[watermarking_mask] = gt_patch[watermarking_mask].clone()
        init_latents_w = torch.fft.ifft2(torch.fft.ifftshift(init_latents_w_fft, dim=(-1, -2))).real
        return init_latents_w
    if 'complex2' == w_injection:
        init_latents_w_fft = torch.fft.fft2(init_latents_w)
        init_latents_w_fft[watermarking_mask] = gt_patch[watermarking_mask].clone()
        init_latents_w = torch.fft.ifft2(init_latents_w_fft).real
        return init_latents_w
    if 'seed' in w_injection:
        if init_latents_w.dtype != gt_patch.dtype:
            gt_patch = gt_patch.to(init_latents_w.dtype)
        init_latents_w[watermarking_mask] = gt_patch[watermarking_mask].clone()
        return init_latents_w

def eval_watermark_single(reversed_latents_no_w, watermarking_mask, gt_patch, w_measurement, w_channel):
    if 'complex' in w_measurement and 'complex2' not in w_measurement:
        reversed_latents_no_w_fft = torch.fft.fftshift(torch.fft.fft2(reversed_latents_no_w), dim=(-1, -2))
        target_patch = gt_patch
    elif 'complex2' in w_measurement:
        reversed_latents_no_w_fft = torch.fft.fft2(reversed_latents_no_w)
        target_patch = gt_patch
    elif 'seed' in w_measurement:
        reversed_latents_no_w_fft = reversed_latents_no_w
        target_patch = gt_patch
    else:
        raise NotImplementedError(f'w_measurement: {w_measurement}')
    eval_results = {}
    if 'l1' in w_measurement:
        masked_tensor = torch.abs(reversed_latents_no_w_fft[watermarking_mask] - target_patch[watermarking_mask])
        if masked_tensor.numel() > 0:
            mask_l1diff_mean = masked_tensor.mean().item()
            mask_l1diff_sum = torch.sum(masked_tensor).item()
        else:
            # Randomly select a value from reversed_latents_no_w_fft if mask is empty
            random_index = random.randint(0, reversed_latents_no_w_fft.numel() - 1)
            random_value = reversed_latents_no_w_fft.view(-1)[random_index]
            mask_l1diff_mean = random_value.item()
            mask_l1diff_sum = random_value.item()
        eval_results['mask_l1diff_mean'] = mask_l1diff_mean
        eval_results['mask_l1diff_sum'] = mask_l1diff_sum

        mask_l1_norm_tensor = torch.abs(reversed_latents_no_w_fft[watermarking_mask])
        if mask_l1_norm_tensor.numel() > 0:
            eval_results['mask_l1_norm'] = torch.sum(mask_l1_norm_tensor).item()
        else:
            random_value = reversed_latents_no_w_fft.view(-1)[random.randint(0, reversed_latents_no_w_fft.numel() - 1)]
            eval_results['mask_l1_norm'] = random_value.item()
    elif 'p_value' in w_measurement:
        test_num = torch.abs(reversed_latents_no_w_fft[watermarking_mask])
        if test_num.numel() > 0:
            eval_results['p_value'] = get_p_value_single(reversed_latents_no_w_fft, watermarking_mask, target_patch, w_measurement)
        else:
            eval_results['p_value'] = 99.9
    else:
        raise NotImplementedError(f'w_measurement: {w_measurement}')

    return eval_results

def get_p_value_single(reversed_latent, watermarking_mask, gt_patch, w_measurement):
    # the reversed_latent has already been transformed if using fourier
    reversed_latents_flatten = reversed_latent[watermarking_mask].flatten()
    target_patch = gt_patch[watermarking_mask].flatten()

    if not 'complex' in w_measurement:
        sigma_wm = reversed_latents_flatten.std()
        lambda_wm = (target_patch ** 2 / sigma_wm ** 2).sum().item()
        x_no_w = (((reversed_latents_flatten - target_patch) / sigma_wm) ** 2).sum().item()
        p_value = scipy.stats.ncx2.cdf(x=x_no_w, df=len(target_patch), nc=lambda_wm)
    else:
        reversed_latents_flatten = torch.concatenate([reversed_latents_flatten.real, reversed_latents_flatten.imag])
        target_patch = torch.concatenate([target_patch.real, target_patch.imag])
        sigma_wm = reversed_latents_flatten.std()
        lambda_wm = (target_patch ** 2 / sigma_wm ** 2).sum().item()
        x_no_w = (((reversed_latents_flatten - target_patch) / sigma_wm) ** 2).sum().item()
        p_value = scipy.stats.ncx2.cdf(x=x_no_w, df=len(target_patch), nc=lambda_wm)

    return p_value

def get_roc_auc(no_w_metrics, w_metrics, w_measurement, name, dir_path):
    preds = no_w_metrics + w_metrics

    t_labels = [0] * len(no_w_metrics) + [1] * len(w_metrics)

    fpr, tpr, thresholds = metrics.roc_curve(t_labels, preds, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    acc = np.max(1 - (fpr + (1 - tpr))/2)
    low = tpr[np.where(fpr<.01)[0][-1]]

    with open(dir_path, 'a') as f:
        f.write(f'='*20)
        f.write('ROC & AUC\n')
        f.write(f'{name}\n')
        f.write(f'auc: {auc}, acc: {acc}, TPR@1%FPR: {low}\n')

def get_metrics(eval_results, w_measurement):
    if 'l1' in w_measurement:
        return -eval_results['mask_l1diff_mean']
    elif 'p_value' in w_measurement:
        return -eval_results['p_value']
    else:
        raise NotImplementedError('w_measurement not implemented!')

def record_results(output_txt_file, eval_results, this_string):
    with open(output_txt_file, 'a') as txt_f:
        txt_f.write(f'='*15)
        txt_f.write(f'{this_string}\n')
        for key, value in eval_results.items():
            txt_f.write(f'{key}: {value}\n')
