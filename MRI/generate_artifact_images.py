# Author: Sayantan Bhadra
# Date: 05/03/2020
# Script for generating noiseless single-coil MR measurements and corresponding pseudoinverse reconstructions with artifacts

import numpy as np
import numpy.fft as fft
import matplotlib
import matplotlib.pyplot as plt

# Load the array of images and select a single slice as ground truth f
dim = 256 # Image dimensions: dimxdim
images = np.load('../axial_batch2_256x256.npy')
f = images[200,:,:]

# Load the undersampling mask
mask = np.fromfile('mask_4_fold_cartesian.dat',np.int32)
mask = mask.reshape(dim,dim)
plt.figure(1); plt.imshow(mask,cmap='gray'); plt.title('Undersampling mask')
plt.show()
mask = fft.ifftshift(mask) # Since FFT is not centered we can shift the mask itself

# Perform forward operation (FFT followed by undersampling)
# The measurement data 'g' in MRI is called the 'k-space'
g = mask * fft.fft2(f)

# The MP pseudoinverse is the IFFT (f_MP)
# Take the real component as FFT and IFFT will introduce imaginary components
f_MP = fft.ifft2(g); f_MP = np.real(f_MP)

plt.figure(2)
plt.subplot(121);plt.imshow(f,cmap='gray');plt.colorbar();plt.title('Ground truth')
plt.subplot(122);plt.imshow(f_MP,cmap='gray');plt.colorbar();plt.title('Reconstructed pseudoinverse image with artifacts')

plt.show()
