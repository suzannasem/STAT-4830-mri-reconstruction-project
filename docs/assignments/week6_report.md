## Brief Summary

* **Zero-filled FFT as a base case for comparison across reconstructions.** The zero-filled FFT has a MSE of 0.010049 and PSNR of 19.98dB, while our previous reconstruction using Gaussian kernels yielded a PSNR of 22.78 dB and an MSE of 0.00529, confirming that Gaussian kernels are an improvement from the base case.
* **LaPlacian Kernel Basis**: We used a LaPlacian kernel basis to generate sharper image lines and found that with a PSNR of 17.30 dB and a MSE of 0.0186, the LaPlacian reconstruction was lower quality than the baseline.
* **Neural Network Reconstruction**: We trained a neural network on undersampled image slices from the same MRI stack and left out the middle image (slice 61). We found a baseline PSNR of the undersampled slice to be 20.0367 dB before a neural network predicted the rest of the image. The neural network reconstruction had a PSNR of 20.5625 dB for an improvement of 0.5258 dB, better than the zero-filled FFT reconstruction, the LaPlacian kernel basis, and the baseline undersampled slice, but not the Gaussian kernel reconstruction.

## Baseline Comparisons

We use the zero-filled Fourier transform as our base case for comparison of reconstruction quality. This transform assigns a value of ‘0,’ meaning no signal is being measured, to all missing values. The zero-filled FFT has a MSE of 0.010049 and PSNR of 19.98dB, while using Gaussian kernels in the reconstruction yielded a PSNR of 22.78 dB and an MSE of 0.00529, confirming that Gaussian kernels are an improvement from the base case. See Figure 1 for a visual comparison.

## Status on Kernels

We show the image reconstruction using a Laplacian kernel basis. Laplacian kernels have sharper boundaries which may be useful for MRI images, where irregularities can be as small as a few pixels. We repeat our simulation using the same dataset and mathematical construction, but restricting the reconstruction to using LaPlacian kernel basis functions. We can visually see that the LaPlacian reconstruction is much lower quality than the zero-filled FFT image. The LaPlacian reconstruction has a PSNR of 17.30 dB and a MSE of 0.0186, meaning it is not an improvement from the zero-filled FFT.

<img width="1415" height="452" alt="download" src="https://github.com/user-attachments/assets/943682b0-dba0-4b7f-ad16-78ab7bf4064e" />

Figure 1. Target Image (left), Reconstruction using LaPlacian Kernels (middle) and zero-filled FFT reconstruction (right).

## Neural Networks & Comparison
As a comparison to the kernel implementation, we train a neural network to map an undersampled image obtained from inverse FFT of k-space to the ground truth image. Thus, the neural network learns a mapping from the corrupted images to clean images from the data (i.e.  where  is the undersampled image and  is the ground truth image). Using our success metric of PSNR, we compute a baseline PSNR, which is the PSNR of the undersampled image to the target, ground truth image. 
An MRI series of 152 slices was loaded and the same slice that was used for the kernel reconstruction (slice 61) was held out from the training set. Then, we undersampled the remaining 119 images and trained a small residual CNN on the undersampled images. We tested on slice 61 in order to compare the neural network performance to the kernel method performance. The held out slice had a baseline PSNR of 20.0367 dB. When testing on the same slice, the neural network reconstruction had a PSNR of 20.5625 dB for an improvement of 0.5258 dB. Thus, the neural network does generalize and improves the baseline, the zero-filled FFT reconstruction, and the kernel reconstruction using the Laplace kernel, but it is very minimal. 
Below is a screenshot of the neural network reconstruction compared to the baseline and ground truth image:

<img width="950" height="336" alt="download" src="https://github.com/user-attachments/assets/4ee4dfb8-97e2-4e4a-bf41-431176ac20d7" />
Figure 2. Target Image (left), undersampled image (middle) and neural network output (right) 

## Next Steps

**Combining kernel reconstruction with diffusion models:** We can use diffusion models to learn the residual of what the kernel reconstruction missed. This is a much simpler model than reconstructing the whole image because the kernels already do a decent job with the reconstruction. Overall, it is a much easier learning problem because the diffusion model only has to learn fine details and high-frequency corrections, which dramatically reduces the difficulty. 
