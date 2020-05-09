# Author: Sayantan Bhadra
# Date: 05/03/2020
# Script for generating noiseless single-coil MR measurements and corresponding pseudoinverse reconstructions with artifacts
import numpy as np
import numpy.fft as fft
import matplotlib
import matplotlib.pyplot as plt

docker = True
# Load the array of images and select a single slice as ground truth f
dim = 256 # Image dimensions: dimxdim
dataset_folder = '/data/datasets/MRI'
images = np.load(dataset_folder + '/axial_batch2_256x256.npy')
f_MP_list = []
for i in range(images.shape[0]):
	# Load the undersampling mask
	mask = np.fromfile('./mask_4_fold_cartesian.dat',np.int32)
	mask = mask.reshape(dim,dim)
	mask = fft.ifftshift(mask) # Since FFT is not centered we can shift the mask itself
	plt.clf()
	plt.figure(1); plt.imshow(mask,cmap='gray'); plt.title('Undersampling mask')
	plt.show()
	plt.savefig('/data/datasets/MRI/mask_image.png')

	# Generate the artifact images
	f = images[200,:,:]
	# Perform forward operation (FFT followed by undersampling)
	# The measurement data 'g' in MRI is called the 'k-space'
	g = mask * fft.fft2(f)
	# The MP pseudoinverse is the IFFT (f_MP)
	# Take the real component as FFT and IFFT will introduce imaginary components
	mean = [0, 0]
	cov = [[np.real(np.max(g))*10, 0],[0, np.real(np.max(g))*10]]
	z = np.squeeze(np.random.multivariate_normal(mean, cov, (dim, dim)).view(np.complex128))
# 	f_MP = fft.ifft2(g); f_MP = np.real(f_MP)
	g1 = g + z
	f_MP1 = fft.ifft2(g1); f_MP1 = np.real(f_MP1)
	f_MP_list.append(f_MP1)

# plt.close('all'); plt.figure(2); plt.clf()
# plt.subplot(321);plt.imshow(f,cmap='gray');plt.colorbar();plt.title('Ground truth')
# plt.subplot(322);plt.imshow(mask,cmap='gray');plt.colorbar();plt.title('Mask')
# plt.subplot(323);plt.imshow(np.real(g),cmap='gray');plt.colorbar();plt.title('Noiseless')
# plt.subplot(324);plt.imshow(f_MP,cmap='gray');plt.colorbar();plt.title('Noiseless Recon')
# plt.subplot(325);plt.imshow(np.real(g1),cmap='gray');plt.colorbar();plt.title('Noisy')
# plt.subplot(326);plt.imshow(f_MP1,cmap='gray');plt.colorbar();plt.title('Noisy Recon')
# plt.tight_layout()
# plt.savefig('/data/datasets/MRI/artifact_image.png')

# save data
f_MP_arr = np.array(f_MP_list, dtype = np.float32)
np.save(dataset_folder + '/axial_batch2_256x256_artifact_noisy.npy', f_MP_arr)

# visualize one group of examples
def save_recon_images(img_file_name, imgs, recons, fig_size):
	from matplotlib.backends.backend_agg import FigureCanvasAgg
	from matplotlib.figure import Figure
	imgs, recons = np.squeeze(imgs), np.squeeze(recons)
	test_size = imgs.shape[0]
	indxs = np.random.randint(0,int(test_size),3)
# 	fig_size = (8,6)
	fig_size = fig_size
	fig = Figure(figsize=fig_size)
	rows, cols = 2, 3
	ax = fig.add_subplot(rows, cols, 1); cax=ax.imshow(imgs[indxs[0],:],cmap='gray'); fig.colorbar(cax); ax.set_title('Image-{}'.format(indxs[0])); ax.set_ylabel('f') 
	ax = fig.add_subplot(rows, cols, 2); cax=ax.imshow(imgs[indxs[1],:],cmap='gray'); fig.colorbar(cax); ax.set_title('Image-{}'.format(indxs[1]));
	ax = fig.add_subplot(rows, cols, 3); cax=ax.imshow(imgs[indxs[2],:],cmap='gray'); fig.colorbar(cax); ax.set_title('Image-{}'.format(indxs[2]));
	ax = fig.add_subplot(rows, cols, 4); cax=ax.imshow(recons[indxs[0],:],cmap='gray'); fig.colorbar(cax); ax.set_ylabel('f_MP')
	ax = fig.add_subplot(rows, cols, 5); cax=ax.imshow(recons[indxs[1],:],cmap='gray'); fig.colorbar(cax);
	ax = fig.add_subplot(rows, cols, 6); cax=ax.imshow(recons[indxs[2],:],cmap='gray'); fig.colorbar(cax);
	canvas = FigureCanvasAgg(fig)
	canvas.print_figure(img_file_name, dpi=100)

save_recon_images(dataset_folder+'/artifact_noisy_examples.png', images, f_MP_arr, fig_size = [11, 5])

