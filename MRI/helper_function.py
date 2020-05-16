import numpy as np
import os
import glob
from natsort import natsorted
from termcolor import colored

def plot_image_pair(img_file_name, img_arr1, img_arr2, fig_size):
	from matplotlib.backends.backend_agg import FigureCanvasAgg
	from matplotlib.figure import Figure
	imgs, imgs2 = img_arr1, img_arr2
	if len(imgs.shape) > 2:
		test_size = imgs.shape[0]
	else:
		test_size = 1
	indx = np.random.randint(0,test_size)
	img1, img2 = imgs[indx,:], imgs2[indx,:]
	fig_size = fig_size
	fig = Figure(figsize=fig_size)
	ax = fig.add_subplot(1, 2, 1); cax=ax.imshow(img1,cmap='gray'); fig.colorbar(cax); ax.set_title('Image')
	ax = fig.add_subplot(1, 2, 2); cax=ax.imshow(img2,cmap='gray'); fig.colorbar(cax); ax.set_title('f_meas+f_null')
	canvas = FigureCanvasAgg(fig)
	canvas.print_figure(img_file_name, dpi=100)

def normalize_0_1(data):
	data = np.squeeze(data)
	shp = data.shape
	_shp = (shp[0],)
	for i in range(1,len(shp)):
		_shp = _shp + (1,)
	data = (data - np.amin(np.amin(data, axis = -1), axis = -1).reshape(_shp))/\
			(np.amax(np.amax(data, axis = -1), axis = -1).reshape(_shp)-\
			np.amin(np.amin(data, axis = -1), axis = -1).reshape(_shp))
	image_sum = np.squeeze(np.apply_over_axes(np.sum, data, axes = [1,2]))
	return data[~np.isnan(image_sum),:]

def pad_128(data):
	return np.pad(data, ((0,0),(10,9),(10,9)), 'mean')

def print_yellow(str):
	from termcolor import colored 
	print(colored(str, 'yellow'))

def print_red(str):
	from termcolor import colored 
	print(colored(str, 'red'))

def print_green(str):
	from termcolor import colored 
	print(colored(str, 'green'))

def print_block(symbol = '*', nb_sybl = 70):
	print_green(symbol*nb_sybl)

def plot_LOSS(file_name, skip_points, train_loss_list, val_loss_list, norm_loss_list, abnorm_loss_list):
	import matplotlib.pyplot as plt
	from matplotlib.backends.backend_agg import FigureCanvasAgg
	from matplotlib.figure import Figure
	fig_size = (8,6)
	fig = Figure(figsize=fig_size)
	file_name = file_name
	ax = fig.add_subplot(111)
	if len(train_loss_list) < skip_points:
		return
	ax.plot(train_loss_list[skip_points:])
	ax.plot(val_loss_list[skip_points:])
	ax.plot(norm_loss_list[skip_points:])
	ax.plot(abnorm_loss_list[skip_points:])
	title = os.path.basename(os.path.dirname(file_name))
	ax.set_title(title)
	ax.set_xlabel('Iterations/100')
	ax.set_ylabel('MSE')
	ax.legend(['Train','Valid','T-norm', 'T-Abnorm'])
	ax.set_xlim([0,len(train_loss_list)])
	canvas = FigureCanvasAgg(fig)
	canvas.print_figure(file_name, dpi=100)

# visualize one group of examples
def save_recon_images_1(img_file_name, imgs, recons, fig_size):
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

def plot_hist(file_name, x, y):
	import matplotlib.pyplot as plt
	from matplotlib.backends.backend_agg import FigureCanvasAgg
	from matplotlib.figure import Figure
	kwargs = dict(alpha=0.6, bins=100, density= False, stacked=True)
	fig_size = (8,6)
	fig = Figure(figsize=fig_size)
	file_name = file_name
	ax = fig.add_subplot(111)
# 	x = np.exp(-x)
# 	y = np.exp(-y)
	ax.hist(x, **kwargs, color='g', label='Norm')
	ax.hist(y, **kwargs, color='r', label='Anomaly')
	title = os.path.basename(os.path.dirname(file_name))
	ax.set_title(title)
	ax.set_xlabel('Error')
	ax.set_ylabel('Frequency')
	ax.legend(['Norm', 'Anomaly'])
	ax.set_xlim([np.min(np.concatenate([x,y])), np.max(np.concatenate([x,y]))])
	canvas = FigureCanvasAgg(fig)
	canvas.print_figure(file_name, dpi=100)

def plot_hist_pixels(file_name, x, y):
	import matplotlib.pyplot as plt
	from matplotlib.backends.backend_agg import FigureCanvasAgg
	from matplotlib.figure import Figure
	kwargs = dict(alpha=0.6, bins=100, density= False, stacked=True)
	fig_size = (8,6)
	fig = Figure(figsize=fig_size)
	file_name = file_name
	ax = fig.add_subplot(111)
	ax.hist(x, **kwargs, color='g', label='Norm')
	ax.hist(y, **kwargs, color='r', label='Anomaly')
	title = os.path.basename(os.path.dirname(file_name))
	ax.set_title(title)
# 	ax.set_xlabel('Error')
	ax.set_xlabel('Mean of normalized pixel values')
	ax.set_ylabel('Frequency')
	ax.legend(['Norm', 'Anomaly'])
	ax.set_xlim([np.min(np.concatenate([x,y])), np.max(np.concatenate([x,y]))])
	canvas = FigureCanvasAgg(fig)
	canvas.print_figure(file_name, dpi=100)

def plot_AUC(file_name, auc_list):
	from matplotlib.backends.backend_agg import FigureCanvasAgg
	from matplotlib.figure import Figure
	fig_size = (8,6)
	fig = Figure(figsize=fig_size)
	file_name = file_name
	ax = fig.add_subplot(111)
	ax.plot(auc_list)
	title = os.path.basename(os.path.dirname(file_name))
	ax.set_title(title)
	ax.set_xlabel('Interations/100')
	ax.set_ylabel('AUC')
	ax.legend(['Detection'])
	ax.set_xlim([0,len(auc_list)])
	canvas = FigureCanvasAgg(fig)
	canvas.print_figure(file_name, dpi=100)

def save_recon_images(img_file_name, imgs, recons, errs, fig_size):
	from matplotlib.backends.backend_agg import FigureCanvasAgg
	from matplotlib.figure import Figure
	imgs, recons, errs = np.squeeze(imgs), np.squeeze(recons), np.squeeze(errs)
	test_size = imgs.shape[0]
	indx = np.random.randint(0,int(test_size/2))
	f, f_MP = imgs[indx,:,:], imgs[int(test_size/2)+indx,:,:]
	f_recon, f_MP_recon = recons[indx,:,:], recons[int(test_size/2)+indx,:,:]
	f_recon_err, f_MP_recon_err = errs[indx,:,:], errs[int(test_size/2)+indx,:,:]
# 	fig_size = (8,6)
	fig_size = fig_size
	fig = Figure(figsize=fig_size)
	rows, cols = 2, 3
	ax = fig.add_subplot(rows, cols, 1); cax=ax.imshow(f,cmap='gray'); fig.colorbar(cax); ax.set_title('Image'); ax.set_ylabel('Normal') 
	ax = fig.add_subplot(rows, cols, 2); cax=ax.imshow(f_recon,cmap='gray'); fig.colorbar(cax); ax.set_title('Recon');
	ax = fig.add_subplot(rows, cols, 3); cax=ax.imshow(f_recon_err,cmap='gray'); fig.colorbar(cax); ax.set_title('Error');
	ax = fig.add_subplot(rows, cols, 4); cax=ax.imshow(f_MP,cmap='gray'); fig.colorbar(cax); ax.set_ylabel('Anomaly')
	ax = fig.add_subplot(rows, cols, 5); cax=ax.imshow(f_MP_recon,cmap='gray'); fig.colorbar(cax);
	ax = fig.add_subplot(rows, cols, 6); cax=ax.imshow(f_MP_recon_err,cmap='gray'); fig.colorbar(cax);
	canvas = FigureCanvasAgg(fig)
	canvas.print_figure(img_file_name, dpi=100)

def generate_folder(folder):
	if not os.path.exists(folder):
		os.system('mkdir -p {}'.format(folder))