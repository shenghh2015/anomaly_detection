# Author: Sayantan Bhadra
# Date: 05/03/2020
# Script for generating noiseless single-coil MR measurements and corresponding pseudoinverse reconstructions with artifacts
# Modified by Shenghua He for generating a large amount of data
import numpy as np
import numpy.fft as fft
import matplotlib
import matplotlib.pyplot as plt

# Load the array of images and select a single slice as ground truth f
dim = 256 # Image dimensions: dimxdim
images = np.load('../axial_batch2_256x256.npy')

# Load the undersampling mask
mask = np.fromfile('mask_4_fold_cartesian.dat',np.int32)
mask = mask.reshape(dim,dim)
mask = fft.ifftshift(mask) # Since FFT is not centered we can shift the mask itself
# plt.figure(1); plt.imshow(mask,cmap='gray'); plt.title('Undersampling mask')
# plt.show()

# Generate the artifact images
# nb_data = 1000
f = images[i,:,:]
# Perform forward operation (FFT followed by undersampling)
# The measurement data 'g' in MRI is called the 'k-space'
g = mask * fft.fft2(f)
# The MP pseudoinverse is the IFFT (f_MP)
# Take the real component as FFT and IFFT will introduce imaginary components
f_MP = fft.ifft2(g); f_MP = np.real(f_MP)

# f_MP_arr = np.array(f_MP_list, dtype = np.float32)

# Save the data
# np.save('./axial_batch2_256x256_artifact.npy', f_MP_arr)
# plt.figure(2)
# plt.subplot(121);plt.imshow(f,cmap='gray');plt.colorbar();plt.title('Ground truth')
# plt.subplot(122);plt.imshow(f_MP,cmap='gray');plt.colorbar();plt.title('Reconstructed pseudoinverse image with artifacts')
# 
# for i in range(20):
# 	idx = np.random.randint(0,999)
# 	plt.imshow(f_MP_arr[idx,:], cmap = 'gray')
# 	plt.pause(1)
# plt.show()
