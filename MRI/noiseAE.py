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
# from models import *
from models2 import auto_encoder

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
parser.add_argument("--train", type=int, default = 65000)
parser.add_argument("--val", type=int, default = 200)
parser.add_argument("--test", type=int, default = 200)
parser.add_argument("--noise_level", type=float, default = 0)
parser.add_argument("--us_factor", type=float, default = 4)
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
train = args.train
val = args.val
test = args.test
noise_level = args.noise_level
us_factor = args.us_factor
version = args.version
loss = args.loss

os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

if docker:
	output_folder = '/data/results/MRI'
else:
	output_folder = './data/MRI'

## model folder
model_name = 'AE{}-{}-cn-{}-fr-{}-ks-{}-bn-{}-skp-{}-res-{}-lr-{}-stps-{}-bz-{}-tr-{}k-vl-{}-test-{}-NL-{}-fact-{}-l-{}'.format(version, os.path.basename(output_folder), nb_cnn, filters, kernel_size, batch_norm, skip, residual, lr, nb_steps, batch_size, int(train/1000), val, test,noise_level, us_factor, loss)
model_folder = os.path.join(output_folder, model_name)
generate_folder(model_folder)

#image size
img_size = 256
## load dataset
print_red('Data loading ...')
X_SA_trn, X_SA_val, X_SA_tst, X_SP_tst = load_MRI_data(docker = docker, train = train, val = val, normal = test, anomaly = test, noise_level = noise_level, us_factor = us_factor)
print_red('Data 0-1 normalization ...')
X_SA_trn, X_SA_val, X_SA_tst, X_SP_tst = normalize_0_1(X_SA_trn), normalize_0_1(X_SA_val), normalize_0_1(X_SA_tst), normalize_0_1(X_SP_tst)
## Dimension adjust
X_SA_trn, X_SA_val, X_SA_tst, X_SP_tst = np.expand_dims(X_SA_trn, axis = 3), np.expand_dims(X_SA_val, axis = 3), np.expand_dims(X_SA_tst, axis = 3),\
		 np.expand_dims(X_SP_tst, axis = 3)
print_red('Data ready!')

# create the graph
scope = 'base'
x = tf.placeholder("float", shape=[None, img_size, img_size, 1])
is_training = tf.placeholder_with_default(False, (), 'is_training')
if version == 1 or version ==2:
	h1, h2, y = auto_encoder(x, nb_cnn = nb_cnn, bn = batch_norm, bn_training = is_training, filters = filters, kernel_size = [kernel_size, kernel_size], scope_name = scope)
elif version == 3:
	h1, h2, y = auto_encoder2(x, nb_cnn = nb_cnn, bn = batch_norm, filters = filters, kernel_size = [kernel_size, kernel_size], scope_name = scope)

# create a saver
key_direct = {}; vars_list = tf.global_variables(scope); key_list = [v.name[:-2] for v in tf.global_variables(scope)]
for key, var in zip(key_list, vars_list):
	key_direct[key] = var
saver = tf.train.Saver(key_direct, max_to_keep=nb_steps)
for v in key_list:
	print_green(v)

if loss == 'mse':
	err_map = tf.square(y - x) 
elif loss == 'correntropy':
	sigma = 0.1
	err_map = -tf.exp(-tf.square(x - y)/sigma)
elif loss == 'bce':
	err_map = -x*tf.log(tf.math.sigmoid(y)) - (1-x)*tf.log(1-tf.math.sigmoid(y))

# loss function
err_mean = tf.reduce_mean(err_map, [1,2,3]); cost = tf.reduce_mean(err_mean)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
	trn_step = tf.train.AdamOptimizer(lr).minimize(cost, var_list= tf.trainable_variables(scope))

# save the results for the methods by use of mean of pixels
Xt = np.expand_dims(np.concatenate([X_SA_tst, X_SP_tst], axis = 0), axis = 3)
img_means = np.squeeze(np.apply_over_axes(np.mean, Xt, axes = [1,2,3]))
yt = np.concatenate([np.zeros((len(X_SA_tst),1)), np.ones((len(X_SP_tst),1))], axis = 0).flatten(); MP_auc = roc_auc_score(yt, img_means)
np.savetxt(os.path.join(model_folder,'MP_stat.txt'), img_means)
plot_hist_pixels(model_folder+'/hist_mean_pixel.png'.format(model_name), img_means[:int(len(img_means)/2)], img_means[int(len(img_means)/2):])

# training
loss_trn_list, loss_val_list, loss_norm_list, loss_anomaly_list, auc_list =[],[],[],[],[]

# nb_steps = 5000
best_loss_val = np.inf
# sess = tf.Session()
with tf.Session() as sess:
	tf.global_variables_initializer().run(session=sess)
	for iteration in range(nb_steps):
		indices = np.random.randint(0, X_SA_trn.shape[0]-1, batch_size)
		# train with batches
		batch_x = X_SA_trn[indices,:]; sess.run(trn_step, feed_dict={x: batch_x,is_training: True})
		if iteration%100 == 0:
			loss_trn = cost.eval(session = sess, feed_dict = {x:batch_x,is_training: False})
			loss_val = cost.eval(session = sess, feed_dict = {x:X_SA_val,is_training: False})
			loss_norm = cost.eval(session = sess, feed_dict = {x:X_SA_tst,is_training: False})
			loss_anomaly = cost.eval(session = sess, feed_dict = {x:X_SP_tst,is_training: False})
			# reconstructed images
			Yn = y.eval(session = sess, feed_dict = {x: X_SA_tst,is_training: False}); Ya = y.eval(session = sess, feed_dict = {x: X_SP_tst, is_training:False})
			y_recon = np.concatenate([Yn, Ya], axis = 0)
			# reconstruction errors-based detection
			norm_err_map = err_map.eval(session = sess, feed_dict = {x: X_SA_tst,is_training: False}); anomaly_err_map = err_map.eval(session = sess, feed_dict = {x: X_SP_tst,is_training: False})
			recon_err_map = np.concatenate([norm_err_map, anomaly_err_map], axis = 0)
			recon_errs = np.apply_over_axes(np.mean, recon_err_map, [1,2,3]).flatten(); AE_auc = roc_auc_score(yt, recon_errs)
			# print out results
			print_block(symbol = '-', nb_sybl = 50)
			print(model_name)
			print_yellow('LOSS: T {0:.4f}, V {1:.4f}, Norm {2:.4f}, Anomaly {3:.4f}; AUC: AE {4:.4f}, M: {5:.4f}; iter {6:}'.\
					format(loss_trn, loss_val, loss_norm, loss_anomaly, AE_auc, MP_auc, iteration))
			# save model
			saver.save(sess, model_folder +'/model', global_step= iteration)
			# save results
			loss_trn_list, loss_val_list, loss_norm_list, loss_anomaly_list, auc_list =\
				np.append(loss_trn_list, loss_trn), np.append(loss_val_list, loss_val),\
					np.append(loss_norm_list, loss_norm), np.append(loss_anomaly_list, loss_anomaly), np.append(auc_list, AE_auc)
			np.savetxt(model_folder+'/train_loss.txt', loss_trn_list); np.savetxt(model_folder+'/val_loss.txt', loss_val_list)
			np.savetxt(model_folder+'/norm_loss.txt', loss_norm_list); np.savetxt(model_folder+'/anomaly_loss.txt',loss_anomaly_list)
			plot_LOSS(model_folder+'/loss-{}.png'.format(model_name), 0, loss_trn_list, loss_val_list, loss_norm_list, loss_anomaly_list)
			np.savetxt(model_folder+'/AE_auc.txt', auc_list); plot_AUC(model_folder+'/auc-{}.png'.format(model_name), auc_list)

			if best_loss_val > loss_val:
				best_loss_val = loss_val
				saver.save(sess, model_folder +'/best'); print_red('update best:{}'.format(model_name))
				np.savetxt(model_folder+'/AE_stat.txt', recon_errs); np.savetxt(model_folder+'/best_auc.txt',[AE_auc, MP_auc])
				plot_hist(model_folder+'/hist-{}.png'.format(model_name), recon_errs[:int(len(recon_errs)/2)], recon_errs[int(len(recon_errs)/2):])
				save_recon_images(model_folder+'/recon-{}.png'.format(model_name), Xt, y_recon, recon_err_map, fig_size = [11,5])