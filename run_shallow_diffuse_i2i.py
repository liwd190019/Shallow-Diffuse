import argparse
import wandb
import copy
from tqdm import tqdm
from statistics import mean, stdev
import os

import torch
import asyncio
import itertools

from inverse_stable_diffusion import *
from diffusers import DDIMScheduler
import open_clip
from optim_utils import *
from io_utils import *
from attackers import *
from arguments import parse_args

import numpy as np
from datetime import datetime

async def diff_watermark(idx, edit_timestep, seed, output_folder_timestep, x0_gt, gt_patch, null_embeddings, device, pipe, args):
    set_random_seed(seed)
    final_result = {}
    output_dir_path = f'{output_folder_timestep}/img{idx}'
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)
    output_txt_file = f'{output_dir_path}/distance.txt'
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(output_txt_file, 'a') as txt_f:
        txt_f.write(f'{current_time}\n')

    x0_gt.save(f'{output_dir_path}/x0_gt.png')
    x0_gt_tensor = transform_img(x0_gt).unsqueeze(0).to(null_embeddings.dtype).to(device)

    x0_no_w_latent_groundtrue = pipe.get_image_latents(x0_gt_tensor, sample=False) # decode image

    xt_no_w_latent = pipe.backward_diffusion(
        latents = x0_no_w_latent_groundtrue,
        text_embeddings=null_embeddings,
        text_embeddings_null=null_embeddings,
        guidance_scale=1.0,
        num_inference_steps = args.num_inference_steps,
        reverse_process = True,
        start_timestep=0,
        end_timestep=edit_timestep,
    )

    # inject watermark
    xt_w_latent = copy.deepcopy(xt_no_w_latent)
    watermarking_mask = get_watermarking_mask(
        init_latents_w = xt_w_latent,
        w_mask_shape=args.w_mask_shape,
        w_radius=args.w_radius, 
        w_channel=args.w_channel,
        device=device)
    watermarking_mask_eval = watermarking_mask.clone()

    xt_w_latent = inject_watermark(xt_w_latent, watermarking_mask, gt_patch, w_injection=args.w_injection)

    x0_no_w_latent = pipe.backward_diffusion(
        latents = xt_no_w_latent,
        text_embeddings=current_prompt_embeddings,
        text_embeddings_null= null_embeddings,
        guidance_scale=1.0,
        num_inference_steps = args.num_inference_steps,
        reverse_process = False,
        start_timestep=args.num_inference_steps - edit_timestep,
        end_timestep=-1,
    )
    x0_no_w_img = pipe.numpy_to_pil(pipe.decode_latents(x0_no_w_latent))[0]
    # store_pil_image(x0_no_w_img, f'{output_dir_path}/x0_no_w.png')

    x0_w_latent = pipe.backward_diffusion(
        latents = xt_w_latent,
        text_embeddings=current_prompt_embeddings,
        text_embeddings_null= null_embeddings,
        guidance_scale=1.0,
        num_inference_steps = args.num_inference_steps,
        reverse_process = False,
        start_timestep=args.num_inference_steps - edit_timestep,
        end_timestep=-1,
    )
    x0_w_img = pipe.numpy_to_pil(pipe.decode_latents(x0_w_latent))[0]
    # store_pil_image(x0_w_img, f'{output_dir_path}/x0_w.png')

    averaged_latent = copy.deepcopy(x0_w_latent)

    for channel_idx in range(4):
        if channel_idx != args.w_channel:
            averaged_latent[:, channel_idx, :, :] = averaged_latent[:, channel_idx, :, :] + (x0_no_w_latent[:, channel_idx, :, :] - averaged_latent[:, channel_idx, :, :]) * 1.0
    averaged_image = pipe.numpy_to_pil(pipe.decode_latents(averaged_latent))[0]
    # store_pil_image(averaged_image, f'{output_dir_path}/averaged_image.png')

    no_w_img_class = one_image(clear_img = x0_no_w_img, label = 'no_w')
    avg_img_class = one_image(averaged_image, 'avg')
    img_class_list = [
        no_w_img_class,
        avg_img_class,
    ]

    attackers = {
        'none': image_distortion_none,
        'jpeg': image_distortion_jpeg,
        'gaussianblur': image_distortion_gaussianblur,
        'gaussianstd': image_distortion_gaussianstd,
        'colorjitter': image_distortion_colorjitter,
        'randomdrop': image_distortion_randomdrop,
        'saltandpepper': image_distortion_saltandpepper,
        'resizerestore': image_distortion_resizerestore,
        'vaebmshj': image_distortion_vae1,
        'vaecheng': image_distortion_vae2,
        'diff': image_distortion_diff,
        'medianblur': image_distortion_medianblur,
        'diffpure': image_distortion_diffpure,
    }

    for attacker_name, attacker in attackers.items():
        attacker(img_class_list, 42, args)
        for each_img_class in img_class_list:
            preprocessed_img = transform_img(getattr(each_img_class, attacker_name)).unsqueeze(0).to(null_embeddings.dtype).to(device)
            image_latents = pipe.get_image_latents(preprocessed_img, sample=False)
            reversed_latents = pipe.forward_diffusion(
                latents=image_latents,
                text_embeddings=null_embeddings,
                guidance_scale=1.0,
                num_inference_steps=args.num_inference_steps,
                start_timestep=0,
                end_timestep=edit_timestep,
            )
            with open(output_txt_file, 'a') as txt_f:
                txt_f.write(f'*'*50)
                txt_f.write(f'{attacker_name}\n')

            eval_results = eval_watermark_single(reversed_latents, watermarking_mask_eval, gt_patch, args.w_measurement, args.w_channel)
            record_results(output_txt_file, eval_results, f'{args.w_measurement}_{each_img_class.label}')
            final_result[f'{each_img_class.label}_metrics_{attacker_name}'] = get_metrics(eval_results, args.w_measurement)

    return final_result

async def main(args, edit_t, pipe, scheduler, dataset, prompt_key, ref_model, ref_clip_preprocess, ref_tokenizer):
    edit_timestep = int(edit_t * args.num_inference_steps)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pipe = pipe.to(device)

    tester_prompt = '' # assume at the detection time, the original prompt is unknown
    null_embeddings = pipe.get_text_embedding(tester_prompt)

    final_result_list = {}
    prefixes = ['no_w_metrics', 'avg_metrics']
    suffixes = ['none', 'jpeg', 'gaussianblur', 'gaussianstd', 'colorjitter','randomdrop', 'saltandpepper', 'resizerestore','vaebmshj', 'vaecheng', 'diff']

    final_result_list.update({f'{prefix}_{suffix}': [] for prefix in prefixes for suffix in suffixes})

    output_folder_timestep = f'output/{args.run_name}/timestep{edit_timestep}'
    if not os.path.exists(output_folder_timestep):
        os.makedirs(output_folder_timestep)
    with open(os.path.join('output', args.run_name, 'config.log'), 'a') as txt_f:
        txt_f.write('the configuration for this run is:\n')
        txt_f.write(f'{args}\n')

    total_tasks = []
    gt_patch = get_watermarking_pattern(pipe, args, device)

    for i in tqdm(range(args.start, args.end)):
        img_name = f'{i}.png'
        x0_gt = Image.open(os.path.join(args.dataset, img_name))
        seed = i + args.w_seed

        diff_watermark_task = asyncio.create_task(diff_watermark(
            idx = i,
            edit_timestep = edit_timestep,
            seed = seed,
            output_folder_timestep = output_folder_timestep,
            x0_gt = x0_gt,
            gt_patch = gt_patch,
            null_embeddings = null_embeddings,
            device = device,
            pipe = pipe,
            args = args))
        total_tasks.append([diff_watermark_task])
        if len(total_tasks) % 5 == 0 and len(total_tasks) != 0:
            for t in itertools.chain.from_iterable(total_tasks):
                await t
            for single_task in total_tasks:
                final_result = single_task[0].result()
                for key, value in final_result.items():
                    final_result_list[key].append(value)

            total_tasks = []

    if len(total_tasks) != 0:
        for t in itertools.chain.from_iterable(total_tasks):
            await t
        for single_task in total_tasks:
            final_result = single_task[0].result()
            for key, value in final_result.items():
                final_result_list[key].append(value)

    attacker_names = suffixes
    for key, value in final_result_list.items():
        if 'no_w' in key or 'clip' in key or 'clear_img' in key:
            continue
        for attacker_name in attacker_names:
            if attacker_name in key:
                base_result_list = final_result_list[f'no_w_metrics_{attacker_name}']

        get_roc_auc(base_result_list, final_result_list[key], args.w_measurement, key, f'{output_folder_timestep}/overall_scores.txt')

    torch.cuda.empty_cache()

if __name__ == '__main__':
    args = parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    scheduler = DDIMScheduler.from_pretrained(args.model_id, subfolder='scheduler')
    pipe = InversableStableDiffusionPipeline.from_pretrained(
        args.model_id,
        scheduler=scheduler,
        torch_dtype=torch.float16,
        revision='fp16',
        )
    pipe.set_progress_bar_config(disable=True)

    # reference model
    if args.reference_model is not None:
        ref_model, _, ref_clip_preprocess = open_clip.create_model_and_transforms(args.reference_model, pretrained=args.reference_model_pretrain, device=device)
        ref_tokenizer = open_clip.get_tokenizer(args.reference_model)

    dataset, prompt_key = get_dataset(args)
    initialize_attackers(args, device)

    for edit_t in args.edit_time_list:
        asyncio.run(main(args, edit_t=edit_t, pipe=pipe, scheduler=scheduler, dataset=dataset, prompt_key=prompt_key, ref_model = ref_model, ref_clip_preprocess = ref_clip_preprocess, ref_tokenizer = ref_tokenizer))
    print('finished.')
