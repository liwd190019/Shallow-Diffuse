import argparse
from io_utils import list_of_floats

def parse_args():
    parser = argparse.ArgumentParser(description='diffusion watermark')
    parser.add_argument('--run_name', default='test')
    parser.add_argument('--dataset', default='Gustavosta/Stable-Diffusion-Prompts')
    parser.add_argument('--start', default=0, type=int)
    parser.add_argument('--end', default=10, type=int)
    parser.add_argument('--image_length', default=512, type=int)
    parser.add_argument('--model_id', default='stabilityai/stable-diffusion-2-1-base')
    parser.add_argument('--guidance_scale', default=7.5, type=float)
    parser.add_argument('--num_inference_steps', default=50, type=int)
    parser.add_argument('--reference_model', default=None)
    parser.add_argument('--reference_model_pretrain', default=None)

    # watermark
    parser.add_argument('--w_seed', default=42, type=int)
    parser.add_argument('--w_channel', default=0, type=int) # The channel to embed the watermark.
    parser.add_argument('--w_pattern', default='rand') # The pattern of the watermark. Options: '[seed/complex/comoplex2]_[ring/zero/rand]'
    parser.add_argument('--w_mask_shape', default='circle') # The shape of the watermark mask. Options: 'circle', 'square', 'ring', 'whole', outercircle'
    parser.add_argument('--w_radius', default=10, type=int) # The radius of the watermark mask.
    parser.add_argument('--w_measurement', default='l1_complex') # The measurement to calculate the watermark. Options: '[l1/p_value]_[complex/complex2/seed]'.
    parser.add_argument('--w_injection', default='complex') # The method to inject the watermark. Options: 'complex' (low-frequency), 'complex2' (high-frequency), 'seed'.

    # image attackers
    parser.add_argument('--jpeg_ratio', default=25, type=int)
    parser.add_argument('--gaussian_blur_r', default=4, type=int)
    parser.add_argument('--gaussian_std', default=0.1, type=float)
    parser.add_argument('--brightness_factor', default=6, type=float)
    parser.add_argument('--vae_attack_model_name', default='bmshj2018-hyperprior') # vae attacker

    # hyperparameters
    parser.add_argument('--edit_time_list', required=True, type=list_of_floats) # The timestep to embed the watermark.

    args = parser.parse_args()
    return args
