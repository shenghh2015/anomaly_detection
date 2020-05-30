# Author: Sayantan Bhadra
# Date: 05/03/2020
# Script for generating noiseless single-coil MR measurements and corresponding pseudoinverse reconstructions with artifacts
# edit by Shenghua He
# Date: 05/13/2020
import numpy as np
import numpy.fft as fft
import matplotlib
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--us_factor", type = float, default = 1)
parser.add_argument("--noise_level", type = float, default = 10)
args = parser.parse_args()
print(args)

us_factor = args.us_factor
if us_factor == 1 or us_factor == 2 or us_factor == 3 or us_factor == 4:
	us_factor = int(us_factor)
noise_level = args.noise_level
docker = True
# Load the array of images and select a single slice as ground truth f
dim = 256 # Image dimensions: dimxdim
dataset_folder = '/data/datasets/MRI'
mask_folder = dataset_folder + '/anomaly_detection_scripts'
images = np.load(dataset_folder + '/axial_batch2_256x256.npy')
if not us_factor == 1:
	mask_base = np.fromfile(mask_folder +'/mask_{}_fold_cartesian.dat'.format(us_factor), np.int32)
	mask_base = mask_base.reshape(dim,dim)
f_MP_list = []
for i in range(images.shape[0]):	
	# Load the undersampling mask
#	mask = fft.ifftshift(mask_base) # Since FFT is not centered we can shift the mask itself
# 	plt.clf()
# 	plt.figure(1); plt.imshow(mask,cmap='gray'); plt.title('Undersampling mask')
# 	plt.show()
# 	plt.savefig('/data/datasets/MRI/mask_image.png')

	# Generate the artifact images
	f = images[i,:,:]
	# Perform forward operation (FFT followed by undersampling)
	# The measurement data 'g' in MRI is called the 'k-space'
	g = fft.fft2(f)
	# The MP pseudoinverse is the IFFT (f_MP)
	# Take the real component as FFT and IFFT will introduce imaginary components
	cov = [[np.real(np.max(g))*noise_level, 0],[0, np.real(np.max(g))*noise_level]]
	z = np.squeeze(np.random.multivariate_normal([0, 0], cov, (dim, dim)).view(np.complex128))
	if us_factor == 1:
		g = g + z
	else:
		mask = fft.ifftshift(mask_base)
		g = mask * g + z
	f_MP = fft.ifft2(g); f_MP = np.real(f_MP)
	f_MP_list.append(f_MP)

# save data
f_MP_arr = np.array(f_MP_list, dtype = np.float32)
np.save(dataset_folder + '/axial_batch2_256x256_artifact-{}_noise-{}.npy'.format(us_factor, noise_level), f_MP_arr)

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

save_recon_images(dataset_folder+'/artifact-{}_noise-{}_examples.png'.format(us_factor, noise_level), images, f_MP_arr, fig_size = [11, 5])

