# Shallow Diffuse: Robust and Invisible Watermarking through Low-Dimensional Subspaces in Diffusion Models
This code is the official implementation of [Shallow Diffuse Watermarks](https://arxiv.org/abs/2410.21088).

If you have any questions, feel free to reach out ([Issues](https://github.com/liwd190019/Shallow-Diffuse/issues) or <wdli@umich.edu>).

## Abstract
The widespread use of AI-generated content from diffusion models has raised significant concerns regarding misinformation and copyright infringement. Watermarking is a crucial technique for identifying these AI-generated images and preventing their misuse. In this paper, we introduce **Shallow Diffuse**, a new watermarking technique that embeds robust and invisible watermarks into diffusion model outputs. Unlike existing approaches that integrate watermarking throughout the entire diffusion sampling process, **Shallow Diffuse** decouples these steps by leveraging the presence of a low-dimensional subspace in the image generation process. This method ensures that a substantial portion of the watermark lies in the null space of this subspace, effectively separating it from the image generation process. Our theoretical and empirical analyses show that this decoupling strategy greatly enhances the consistency of data generation and the detectability of the watermark. Extensive experiments further validate that our **Shallow Diffuse** outperforms existing watermarking methods in terms of robustness and consistency.


## Method
<img src=assets/method.jpg  width="100%" height="100%">

**Shallow Diffuse** builds on Tree-Ring<sub>[1]</sub> to embed watermarks within the low-dimensional subspace of a diffusion model<sub>[2-4]</sub>, enabling training-free server-scenario and user-scenario watermarking while significantly improving image consistency and maintaining robustness against adversarial attacks.


[1] Wen, Y., Kirchenbauer, J., Geiping, J., & Goldstein, T. (2024). Tree-rings watermarks: Invisible fingerprints for diffusion images. Advances in Neural Information Processing Systems, 36. \
[2] Stanczuk, J. P., Batzolis, G., Deveney, T., & Schönlieb, C. B. Diffusion Models Encode the Intrinsic Dimension of Data Manifolds. In Forty-first International Conference on Machine Learning. \
[3] Chen, S., Zhang, H., Guo, M., Lu, Y., Wang, P., & Qu, Q. Exploring Low-Dimensional Subspace in Diffusion Models for Controllable Image Editing. In The Thirty-eighth Annual Conference on Neural Information Processing Systems. \
[4] Wang, P., Zhang, H., Zhang, Z., Chen, S., Ma, Y., & Qu, Q. (2024). Diffusion models learn low-dimensional distributions via subspace clustering. arXiv preprint arXiv:2409.02426.


## Dependencies
- PyTorch == 1.13.0
- transformers == 4.23.1
- diffusers == 0.11.1
- datasets

Note: higher diffusers version may not be compatible with the DDIM inversion code.

## Example Usage
```
python run_shallow_diffuse_t2i.py   \
    --run_name test                 \
    --w_pattern ring                \
    --start 0                       \
    --end 1000                      \
    --reference_model ViT-g-14      \
    --reference_model_pretrain laion2b_s12b_b42k   \
    --w_channel 3                   \
    --w_pattern complex2_ring       \
    --w_mask_shape circle           \
    --w_radius 10                   \
    --w_measurement l1_complex2     \
    --w_injection  complex2         \
    --edit_time_list 0.3
```

## Parameters
Please refer to `arguments.py`.

## Suggestions and Pull Requests are welcome!

## Credits
- This project is inspired by [Tree-Ring Watermarks](https://github.com/YuxinWenRick/tree-ring-watermark) and [LOCO-Edit](https://github.com/ChicyChen/LOCO-Edit).
