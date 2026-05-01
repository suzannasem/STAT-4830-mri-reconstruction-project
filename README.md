# Exploring Optimization-based MRI and Image Reconstruction Methods with PyTorch

## Abstract

This project evaluates image reconstruction from undersampled k-space measurements, motivated by the need for accelerated MRI acquisition as well as general image reconstruction. We compare various methods, including kernel-based basis functions, Residual CNNs, U-Net, SRCNN, and Diffusion models, across MRI (UPenn-GBM), natural images (Oxford-IIIT Pet), and video (VID4) datasets. Zero-filled FFT provided a strong baseline throughout the project, while zero-filled FFT combined with a residual CNN achieved the best multi-slice MRI results (PSNR 33.20 dB), while multi-frame methods (MFSR-GAN) yielded the highest overall results (38.66 dB PSNR) by leveraging temporal context. We also implemented a self-supervised approach (SSDiffRecon) for reconstruction in environments without ground truth labels.
