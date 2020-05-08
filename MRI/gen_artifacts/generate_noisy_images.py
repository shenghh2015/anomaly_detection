# Author: Sayantan Bhadra
# Date: 05/03/2020
# Script for generating noiseless single-coil MR measurements and corresponding pseudoinverse reconstructions with artifacts
# Modified by Shenghua He for generating a large amount of data
import numpy as np
import numpy.fft as fft
import matplotlib
import matplotlib.pyplot as plt

docker = True
# Load the array of images and select a single slice as ground truth f
dim = 256 # Image dimensions: dimxdim
mean = [0, 0]
cov = [[sigma**2, 0],[0, sigma**2]]
dataset_folder = '/data/datasets/MRI'
images = np.load(dataset_folder + '/axial_batch2_256x256.npy')
# Load the undersampling mask
mask = np.fromfile('./mask_4_fold_cartesian.dat',np.int32)
mask = mask.reshape(dim,dim)
mask = fft.ifftshift(mask) # Since FFT is not centered we can shift the mask itself
# plt.figure(1); plt.imshow(mask,cmap='gray'); plt.title('Undersampling mask')
# plt.show()

# Generate the artifact images
# nb_data = 1000
f_MP_list = []
for i in range(len(images)):
	f = images[i,:,:]
	# Perform forward operation (FFT followed by undersampling)
	# The measurement data 'g' in MRI is called the 'k-space'
	g = mask * fft.fft2(f)
	# The MP pseudoinverse is the IFFT (f_MP)
	# Take the real component as FFT and IFFT will introduce imaginary components
	z = np.squeeze(np.random.multivariate_normal(mean, cov, (dim, dim)).view(np.complex128))
	g = g + z
	f_MP = fft.ifft2(g); f_MP = np.real(f_MP)
	f_MP_list.append(f_MP)

f_MP_arr = np.array(f_MP_list, dtype = np.float32)

# Save the data
np.save(dataset_folder + '/axial_batch2_256x256_artifact_noise_{}.npy'.format(sigma), f_MP_arr)
# plt.figure(2)

## plot figure to show noise
plt.clf()
fig = plt.figure(2)
f = images[500,:,:]
g = mask * fft.fft2(f)
max_g = np.max(np.real(g))
s1, s2, s3, s4, s5 = 0.01, 0.05, 0.10, 0.15, 0.5
cov1, cov2, cov3, cov4, cov5 = [[s1*max_g,0],[0, s1*max_g]], [[s2*max_g,0],[0, s2*max_g]],\
	[[s3*max_g,0],[0, s3*max_g]], [[s4*max_g,0],[0, s4*max_g]], [[s5*max_g,0],[0, s5*max_g]]
z1 = np.squeeze(np.random.multivariate_normal(mean, cov1, (dim, dim)).view(np.complex128))
z2 = np.squeeze(np.random.multivariate_normal(mean, cov2, (dim, dim)).view(np.complex128))
z3 = np.squeeze(np.random.multivariate_normal(mean, cov3, (dim, dim)).view(np.complex128))
z4 = np.squeeze(np.random.multivariate_normal(mean, cov4, (dim, dim)).view(np.complex128))
z5 = np.squeeze(np.random.multivariate_normal(mean, cov4, (dim, dim)).view(np.complex128))
g1, g2, g3, g4, g5 = g + z1, g + z2, g + z3, g + z4, g + z5
f_MP = fft.ifft2(g); f_MP = np.real(f_MP)
f_MP1 = fft.ifft2(g1); f_MP1 = np.real(f_MP1)
f_MP2 = fft.ifft2(g2); f_MP2 = np.real(f_MP2)
f_MP3 = fft.ifft2(g3); f_MP3 = np.real(f_MP3)
f_MP4 = fft.ifft2(g4); f_MP4 = np.real(f_MP4)
f_MP5 = fft.ifft2(g5); f_MP5 = np.real(f_MP5)
# plt.subplot(121);plt.imshow(f,cmap='gray');plt.colorbar();plt.title('Ground truth')
# plt.subplot(122);plt.imshow(f_MP,cmap='gray');plt.colorbar();plt.title('Reconstructed pseudoinverse image with artifacts')
plt.clf()
ax = fig.add_subplot(2,3,1); cax = ax.imshow(f, cmap ='gray'); fig.colorbar(cax); ax.set_title('Ground truth')
ax = fig.add_subplot(2,3,2); cax = ax.imshow(f_MP1, cmap ='gray'); fig.colorbar(cax); ax.set_title('MP-noise:1')
ax = fig.add_subplot(2,3,3); cax = ax.imshow(f_MP2, cmap ='gray'); fig.colorbar(cax); ax.set_title('MP-noise:2')
ax = fig.add_subplot(2,3,4); cax = ax.imshow(f_MP3, cmap ='gray'); fig.colorbar(cax); ax.set_title('MP-noise:3')
ax = fig.add_subplot(2,3,5); cax = ax.imshow(f_MP4, cmap ='gray'); fig.colorbar(cax); ax.set_title('MP-noise:4')
ax = fig.add_subplot(2,3,6); cax = ax.imshow(f_MP5, cmap ='gray'); fig.colorbar(cax); ax.set_title('MP-noise:5')
plt.tight_layout()
fig.savefig(dataset_folder + '/noisy_image.png')
ax = fig.add_subplot(2,3,1); cax = ax.imshow(f, cmap ='gray'); fig.colorbar(cax); ax.set_title('Ground truth')
ax = fig.add_subplot(2,3,2); cax = ax.imshow(images[100,:], cmap ='gray'); fig.colorbar(cax); ax.set_title('MP-noise:1')
ax = fig.add_subplot(2,3,3); cax = ax.imshow(images[200,:], cmap ='gray'); fig.colorbar(cax); ax.set_title('MP-noise:2')
ax = fig.add_subplot(2,3,4); cax = ax.imshow(images[300,:], cmap ='gray'); fig.colorbar(cax); ax.set_title('MP-noise:3')
ax = fig.add_subplot(2,3,5); cax = ax.imshow(images[400,:], cmap ='gray'); fig.colorbar(cax); ax.set_title('MP-noise:4')
ax = fig.add_subplot(2,3,6); cax = ax.imshow(images[500,:], cmap ='gray'); fig.colorbar(cax); ax.set_title('MP-noise:5')
plt.tight_layout()
fig.savefig(dataset_folder + '/artifact_image.png')
# for i in range(20):
# 	idx = np.random.randint(0,999)
# 	plt.imshow(f_MP_arr[idx,:], cmap = 'gray')
# 	plt.pause(1)
# plt.show()
