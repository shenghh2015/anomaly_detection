import tensorflow as tf
import tensorflow.math as tm
import math

import numpy as np
import os
import glob
from natsort import natsorted
from termcolor import colored 
import argparse
from sklearn.metrics import roc_auc_score
import scipy.io
import scipy.misc as misc

from load_data import *
from models import *

## functions
def str2bool(value):
    return value.lower() == 'true'

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
	ax = fig.add_subplot(rows, cols, 1); cax=ax.imshow(f,cmap='gray'); fig.colorbar(cax); ax.set_title('Image'); ax.set_ylabel('f') 
	ax = fig.add_subplot(rows, cols, 2); cax=ax.imshow(f_recon,cmap='gray'); fig.colorbar(cax); ax.set_title('Recon');
	ax = fig.add_subplot(rows, cols, 3); cax=ax.imshow(f_recon_err,cmap='gray'); fig.colorbar(cax); ax.set_title('Error');
	ax = fig.add_subplot(rows, cols, 4); cax=ax.imshow(f_MP,cmap='gray'); fig.colorbar(cax); ax.set_ylabel('f_MP')
	ax = fig.add_subplot(rows, cols, 5); cax=ax.imshow(f_MP_recon,cmap='gray'); fig.colorbar(cax);
	ax = fig.add_subplot(rows, cols, 6); cax=ax.imshow(f_MP_recon_err,cmap='gray'); fig.colorbar(cax);
	canvas = FigureCanvasAgg(fig)
	canvas.print_figure(img_file_name, dpi=100)

def generate_folder(folder):
	if not os.path.exists(folder):
		os.system('mkdir -p {}'.format(folder))

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default = 1)
parser.add_argument("--docker", type = str2bool, default = True)
parser.add_argument("--cn", type=int, default = 4)
parser.add_argument("--fr", type=int, default = 32)
parser.add_argument("--ks", type=int, default = 5)
parser.add_argument("--bn", type=str2bool, default = True)
parser.add_argument("--skp", type=str2bool, default = False)
parser.add_argument("--res", type=str2bool, default = False)
parser.add_argument("--lr", type=float, default = 1e-5)
parser.add_argument("--step", type=int, default = 1000)
parser.add_argument("--bz", type=int, default = 50)
# parser.add_argument("--dataset", type=str, default = 'dense')
parser.add_argument("--train", type=int, default = 65000)
parser.add_argument("--val", type=int, default = 200)
parser.add_argument("--test", type=int, default = 200)
parser.add_argument("--noise", type=float, default = 0)
parser.add_argument("--version", type=int, default = 1)
parser.add_argument("--loss", type = str, default = 'mse')

args = parser.parse_args()
print(args)

gpu = args.gpu
docker = args.docker
nb_cnn = args.cn
filters = args.fr
kernel_size = args.ks
batch_norm = args.bn
skip = args.skp
residual = args.res
lr = args.lr
nb_steps = args.step
batch_size = args.bz
# dataset = args.dataset
train = args.train
val = args.val
test = args.test
noise = args.noise
version = args.version
loss = args.loss

os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

if docker:
	output_folder = '/data/results/MRI'
else:
	output_folder = './data/MRI'

## model folder
model_name = 'f-AE{}-{}-cn-{}-fr-{}-ks-{}-bn-{}-skp-{}-res-{}-lr-{}-stps-{}-bz-{}-tr-{}k-vl-{}-test-{}-n-{}-l-{}'.format(version, os.path.basename(output_folder), nb_cnn, filters, kernel_size, batch_norm, skip, residual, lr, nb_steps, batch_size, int(train/1000), val, test,noise, loss)
model_folder = os.path.join(output_folder, model_name)
generate_folder(model_folder)

#image size
img_size = 256
## load dataset
data_folder = output_folder+'/AE2-MRI-cn-6-fr-32-ks-3-bn-True-skp-False-res-False-lr-0.0001-stps-200000-bz-50-tr-65k-vl-200-test-200-n-40.0-l-correntropy'
X_SA_trn = np.load(data_folder + '/train.npy'); X_SA_val = np.load(data_folder +'/val.npy')
X_SA_tst = np.load(data_folder + '/tst.npy'); X_SP_tst = np.load(data_folder + '/anomaly.npy')
# X_SA_trn, X_SA_val, X_SA_tst, X_SP_tst = load_anomaly_data(dataset = dataset, train = train, valid = 400, test = 400)
# noise = 10
# X_SA_trn, X_SA_val, X_SA_tst, X_SP_tst = load_MRI_true_data(docker = docker, train = train, val = val, normal = test, anomaly = test, noise = noise)
# print_green('Mean: trn {0:.4f}, val {1:.4f}, tst {2:.4f}, anomaly {3:.4f}'.format(np.mean(X_SA_trn), np.mean(X_SA_val), np.mean(X_SA_tst), np.mean(X_SP_tst)))
# print_green('Maxi: trn {0:.4f}, val {1:.4f}, tst {2:.4f}, anomaly {3:.4f}'.format(np.max(X_SA_trn), np.max(X_SA_val), np.max(X_SA_tst), np.max(X_SP_tst)))
# print_green('Mini: trn {0:.4f}, val {1:.4f}, tst {3:.4f}, anomaly {3:.4f}'.format(np.min(X_SA_trn), np.min(X_SA_val), np.min(X_SA_tst), np.min(X_SP_tst)))

# 0-1 normalization
# X_SA_trn, X_SA_val, X_SA_tst, X_SP_tst = normalize_0_1(X_SA_trn), normalize_0_1(X_SA_val), normalize_0_1(X_SA_tst), normalize_0_1(X_SP_tst)
# padding into 128x128 pixels
# X_SA_trn, X_SA_val, X_SA_tst, X_SP_tst = pad_128(X_SA_trn), pad_128(X_SA_val), pad_128(X_SA_tst), pad_128(X_SP_tst)

## test data
Xt = np.concatenate([X_SA_tst, X_SP_tst], axis = 0)
yt = np.concatenate([np.zeros((len(X_SA_tst),1)), np.ones((len(X_SP_tst),1))], axis = 0).flatten()

## Dimension adjust
# X_SA_trn, X_SA_val, X_SA_tst, X_SP_tst, Xt = np.expand_dims(X_SA_trn, axis = 3), np.expand_dims(X_SA_val, axis = 3), np.expand_dims(X_SA_tst, axis = 3),\
# 		 np.expand_dims(X_SP_tst, axis = 3), np.expand_dims(Xt, axis = 3)


# create the graph
scope = 'base'
x = tf.placeholder("float", shape=[None, 256, 256, 1])
if version == 1 or version ==2:
	h1, h2, y = auto_encoder(x, nb_cnn = nb_cnn, bn = batch_norm, filters = filters, kernel_size = [kernel_size, kernel_size], scope_name = scope)
elif version == 3:
	h1, h2, y = auto_encoder2(x, nb_cnn = nb_cnn, bn = batch_norm, filters = filters, kernel_size = [kernel_size, kernel_size], scope_name = scope)

if loss == 'mse':
	err_map = tf.square(y - x) 
elif loss == 'correntropy':
	sigma = 0.1
	err_map = -tf.exp(-tf.square(x - y)/sigma)
elif loss == 'ssim':
	err_map = 1- tf.image.ssim(y, x, max_val = 1.0)
elif loss == 'bce':
	err_map = -x*tf.log(tf.math.sigmoid(y)) - (1-x)*tf.log(1-tf.math.sigmoid(y))

err_mean = tf.reduce_mean(err_map, [1,2,3])
cost = tf.reduce_mean(err_mean)
	
# create a saver
vars_list = tf.trainable_variables(scope)
key_list = [v.name[:-2] for v in tf.trainable_variables(scope)]
key_direct = {}
for key, var in zip(key_list, vars_list):
	key_direct[key] = var
saver = tf.train.Saver(key_direct, max_to_keep=nb_steps)

# print out trainable parameters
for v in key_list:
	print_green(v)

trn_step = tf.train.AdamOptimizer(lr).minimize(cost, var_list= vars_list)

# save the results for the methods by use of mean of pixels
img_means = np.squeeze(np.apply_over_axes(np.mean, Xt, axes = [1,2,3]))
mean_auc = roc_auc_score(yt, img_means)
np.savetxt(os.path.join(model_folder,'Pixel_mean_stat.txt'), img_means)
plot_hist_pixels(model_folder+'/hist_mean_pixel.png'.format(model_name), img_means[:int(len(img_means)/2)], img_means[int(len(img_means)/2):])

# training
trn_err_list = []
val_err_list = []
norm_err_list = []
anomaly_err_list = []
auc_list = []
n_auc_list = []

# nb_steps = 5000
best_val_err = np.inf
# sess = tf.Session()
with tf.Session() as sess:
	tf.global_variables_initializer().run(session=sess)
	for iteration in range(nb_steps):
		indices = np.random.randint(0, X_SA_trn.shape[0]-1, batch_size)
		batch_x = X_SA_trn[indices,:]
		sess.run(trn_step, feed_dict={x: batch_x})
		if iteration%100 == 0:
			trn_err = cost.eval(session = sess, feed_dict = {x:batch_x})
			val_err = cost.eval(session = sess, feed_dict = {x:X_SA_val})
			tst_SA_err = cost.eval(session = sess, feed_dict = {x:X_SA_tst})
			tst_SP_err = cost.eval(session = sess, feed_dict = {x:X_SP_tst})
			y_recon = y.eval(session = sess, feed_dict = {x:Xt})
			tst_pixel_errs = err_map.eval(session = sess, feed_dict = {x:Xt})
			tst_img_errs = np.squeeze(np.apply_over_axes(np.mean, tst_pixel_errs, axes = [1,2,3]))
			# normalized errs
			max_val, min_val = np.max(tst_pixel_errs), np.min(tst_pixel_errs)
			tst_pixel_errs1 = []
			for i in range(tst_pixel_errs.shape[0]):
				err = tst_pixel_errs[i,:]; err = (err -np.min(err))/(np.max(err)-np.min(err))*(max_val -min_val)+min_val 
				tst_pixel_errs1.append(err.reshape(1,256,256,1))
			tst_pixel_errs_n = np.concatenate(tst_pixel_errs1)
			tst_img_errs_n = np.squeeze(np.apply_over_axes(np.mean, tst_pixel_errs_n, axes = [1,2,3]))
			test_auc = roc_auc_score(yt, tst_img_errs); test_auc_n = roc_auc_score(yt, tst_img_errs_n);
			print_block(symbol = '-', nb_sybl = 50)
			print_yellow('RE: train {0:.4f}, val {1:.4f}, normal {2:.4f}, abnormal {3:.4f}; AUC: AE {4:.4f}, AE+normlize {5:.4f}, M: {6:.4f}; iter {7:}'.\
					format(trn_err, val_err, tst_SA_err, tst_SP_err, test_auc, test_auc_n, mean_auc, iteration))
			print(model_name)
			# save model
			saver.save(sess, model_folder +'/model', global_step= iteration)
			# save results
			trn_err_list, val_err_list, norm_err_list, anomaly_err_list, auc_list =\
				np.append(trn_err_list, trn_err), np.append(val_err_list, val_err),\
					np.append(norm_err_list, tst_SA_err), np.append(anomaly_err_list, tst_SP_err), np.append(auc_list, test_auc)
			n_auc_list.append(test_auc_n)
			np.savetxt(os.path.join(model_folder,'train_loss.txt'), trn_err_list)
			np.savetxt(os.path.join(model_folder,'val_loss.txt'), val_err_list)
			np.savetxt(os.path.join(model_folder,'norm_loss.txt'), norm_err_list)
			np.savetxt(os.path.join(model_folder,'anomaly_loss.txt'),anomaly_err_list)
			np.savetxt(os.path.join(model_folder,'test_auc.txt'),auc_list)
			np.savetxt(os.path.join(model_folder,'test_auc_n.txt'),n_auc_list)
			loss_file = os.path.join(model_folder,'loss-{}.png'.format(model_name))
			plot_LOSS(loss_file, 0, trn_err_list, val_err_list, norm_err_list, anomaly_err_list)
			auc_file = os.path.join(model_folder,'auc-{}.png'.format(model_name))
			plot_AUC(auc_file, auc_list); plot_AUC(model_folder+'/n_auc-{}.png'.format(model_name), n_auc_list)
			if best_val_err > val_err:
				best_val_err = val_err
				np.savetxt(os.path.join(model_folder,'AE_stat.txt'), tst_img_errs)
				np.savetxt(os.path.join(model_folder,'AE_n_stat.txt'), tst_img_errs_n)
				np.savetxt(os.path.join(model_folder,'best_auc.txt'),[test_auc, test_auc_n, mean_auc])
				hist_file = os.path.join(model_folder,'hist-{}.png'.format(model_name))
				plot_hist(hist_file, tst_img_errs[:int(len(tst_img_errs)/2)], tst_img_errs[int(len(tst_img_errs)/2):])
				plot_hist(model_folder+'/hist-n-{}.png'.format(model_name), tst_img_errs_n[:int(len(tst_img_errs_n)/2)], tst_img_errs_n[int(len(tst_img_errs_n)/2):])
				saver.save(sess, model_folder +'/best')
				print_red('update best: {}'.format(model_name))
				# save reconstructed images
				img_file_name = os.path.join(model_folder,'recon-{}.png'.format(model_name))
				save_recon_images(img_file_name, Xt, y_recon, tst_pixel_errs, fig_size = [11,5])
				img_file_name = os.path.join(model_folder,'recon-n-{}.png'.format(model_name))
				save_recon_images(img_file_name, Xt, y_recon, tst_pixel_errs_n, fig_size = [11,5])