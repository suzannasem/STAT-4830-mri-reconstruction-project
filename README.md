# Exploring Optimization-based MRI and Image Reconstruction Methods with PyTorch

**STAT 4830 — Spring 2026**  
Aaron Meslin, Amy Zhou, Suzanna Semaan, Sydney Bergstrom

---

## Overview

This project evaluates image reconstruction from undersampled k-space measurements, motivated by the need for accelerated MRI acquisition as well as general image reconstruction. We compare various methods, including kernel-based basis functions, Residual CNNs, U-Net, SRCNN, and Diffusion models, across MRI (UPenn-GBM), natural images (Oxford-IIIT Pet), and video (VID4) datasets. Zero-filled FFT provided a strong baseline throughout the project, while zero-filled FFT combined with a residual CNN achieved the best multi-slice MRI results (PSNR 33.20 dB), while multi-frame methods (MFSR-GAN) yielded the highest overall results (38.66 dB PSNR) by leveraging temporal context. We also implemented a self-supervised approach (SSDiffRecon) for reconstruction in environments without ground truth labels.


---

## Repository Contents

### `General_Image_All_Metrics_SSIM`
Notebook covering reconstruction of natural RGB images from the **Oxford-IIIT Pet Dataset** using masked Fourier coefficients. Experiments include:

- Kernel hyperparameter search (Gaussian vs. Laplacian) optimized over both MSE and SSIM
- Zero-filled FFT baseline
- Residual CNN refinement on top of kernel and zero-filled reconstructions
- Diffusion model correction

Key finding: zero-filled FFT + residual CNN was the strongest approach at **34.18 dB PSNR / SSIM 0.9104**. The diffusion model underperformed the residual CNN on natural images, likely because kernel oversmoothing left less recoverable structure.

---

### `Multi_Slice_All_Metrics_SSIM_Optimized_BUGFIXED`
Notebook for multi-slice MRI reconstruction using the **UPenn-GBM Cancer Imaging Archive**. Covers:

- Multi-slice kernel hyperparameter tuning (kernel type, width, number of kernels, TV regularization weight) optimized over both MSE and SSIM
- Zero-filled FFT baseline across held-out test slices
- Residual CNN corrector applied to kernel and zero-filled FFT reconstructions
- Diffusion model corrector
- Self-supervised diffusion reconstruction (SSDiffRecon, Korkmaz et al. 2024) — trained only on undersampled k-space, without ground truth labels

**Best multi-slice MRI result:** zero-filled FFT + residual CNN at **33.20 dB PSNR / SSIM 0.6758**.

---

### `MFCNN.ipynb`
Implements multi-frame super-resolution on the Vid4 city sequence using two models:

- **MFCNN** (Greaves & Winter, 2016): Concatenates a burst of 5 low-resolution frames along the channel dimension after bicubic upsampling, then learns pixel-wise alignment and fusion via convolutional layers.
- **MFSR-GAN** (Khan et al., CVPRW 2025): Extends MFCNN with reference-difference computation, deformable convolutions for motion alignment, channel attention, and a relativistic GAN loss to recover high-frequency texture.

Pipeline: load HR frames → spatially downsample to form LR burst → reconstruct with MFCNN / MFSR-GAN → evaluate with PSNR and SSIM.

**Best result:** MFSR-GAN achieved **38.66 dB PSNR / SSIM 0.9729** on Vid4, compared to 33.49 dB from a single-frame CNN baseline.

---

## Methods Summary

| Method | Dataset | PSNR | SSIM |
|---|---|---|---|
| Zero-filled FFT (baseline) | MRI | 19.98 dB | — |
| Gaussian Kernel | MRI | 22.78 dB | — |
| Kernel + Residual CNN | MRI | 26.06 dB | 0.4153 |
| Zero-filled FFT + Residual CNN | MRI | **33.20 dB** | 0.6758 |
| Kernel + Diffusion | MRI | 31.61 dB | 0.7796 |
| Zero-filled FFT + Diffusion | MRI | 31.12 dB | 0.7620 |
| SSDiffRecon (self-supervised) | MRI | 26.12 dB | 0.4270 |
| Zero-filled FFT + Residual CNN | Natural Images | **34.18 dB** | 0.9104 |
| Zero-filled FFT + Diffusion | Natural Images | 23.46 dB | 0.4564 |
| MFSR-GAN | Vid4 | **38.66 dB** | 0.9729 |

---

## Key Findings

- **Zero-filled FFT is a surprisingly strong baseline** that kernel methods rarely surpassed on either dataset.
- **Zero-filled FFT + residual CNN** was the best single-image strategy across both MRI and natural images, demonstrating that the pattern generalizes across image types.
- **Multi-frame methods (MFSR-GAN)** yielded the highest overall PSNR by leveraging temporal context across adjacent frames.
- **Diffusion models** worked better for MRI than for natural images; the SSIM-optimized kernel provided a better initialization for the diffusion model than the MSE-optimized kernel.
- **SSDiffRecon** (self-supervised, no ground truth required) performed reasonably given the harder problem, but did not match reported literature results, likely due to training dataset size and compute constraints.
- **Optimization objective matters:** MSE vs. SSIM optimization selects different kernel hyperparameters and can substantially change reconstruction character, especially for SSDiffRecon where SSIM-based training destabilized results.

---

## Datasets

- **UPenn-GBM** — Brain MRI stacks (open-source via NBIA Data Retrieval System)
- **Oxford-IIIT Pet Dataset** — Natural RGB animal images
- **Vid4** — Standard video super-resolution benchmark

---

## Dependencies

```
torch
torchvision
numpy
matplotlib
Pillow
scikit-image
tqdm
```

Install with:
```bash
pip install torch torchvision numpy matplotlib Pillow scikit-image tqdm
```

---

## References

- Dong et al. (2015). *Learning a Deep Convolutional Network for Image Super-Resolution.* (SRCNN)
- Greaves & Winter (2016). *Multi-Frame CNN Super-Resolution.* (MFCNN)
- Khan et al. (CVPRW 2025). *MFSR-GAN.*
- Korkmaz et al. (2024). *SSDiffRecon: Self-Supervised Diffusion Reconstruction.*

