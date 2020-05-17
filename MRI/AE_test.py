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

from load_data import load_MRI_anomaly
from models2 import auto_encoder
from helper_function import normalize_0_1, print_yellow, print_red, print_green, print_block
from helper_function import plot_hist, plot_LOSS, plot_AUC, plot_hist_pixels
from helper_function import generate_folder, save_recon_images

## functions
def str2bool(value):
    return value.lower() == 'true'

gpu = 2; docker = True
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

if docker:
# 	output_folder = '/data/results/MRI/MRI_AE'
	output_folder = '/data/results/MRI/'
else:
	output_folder = './data/MRI'

model_name = 'AE1-MRI-cn-6-fr-32-ks-5-bn-False-lr-0.0001-stps-100000-bz-50-tr-65k-vl-400-test-1000-l-mse'

splits = model_name.split('-')
if len(splits[0])<=2:
	version =1
else:
	version = int(splits[0][2])
for i in range(len(splits)):
	if splits[i] == 'cn':
		nb_cnn = int(splits[i+1])
	elif splits[i] == 'fr':
		filters = int(splits[i+1])
	elif splits[i] == 'bn':
		if splits[i+1]=='True':
			batch_norm = True
		else:
			batch_norm = False
	elif splits[i] == 'ks':
		kernel_size = int(splits[i+1])
	elif splits[i] == 'tr':
		train = int(splits[i+1][:2])* 1000
	elif splits[i] == 'vl':
		val = int(splits[i+1])
	elif splits[i] == 'test':
		test = int(splits[i+1])
	elif splits[i] == 'n' or splits[i]=='NL':
		noise = float(splits[i+1])

model_folder = os.path.join(output_folder, model_name)

## load data
print_red('Data loading ...')
_, _, X_SA_tst, X_SP_tst1 = load_MRI_true_data(docker = docker, train = train, val = val, normal = test, anomaly = test, version = 1)
_, _, _, X_SP_tst2 = load_MRI_true_data(docker = docker, train = train, val = val, normal = test, anomaly = test, version = 2)
_, _, _, X_SP_tst3 = load_MRI_true_data(docker = docker, train = train, val = val, normal = test, anomaly = test, version = 3)
X_SA_tst, X_SP_tst1, X_SP_tst2, X_SP_tst3 = normalize_0_1(X_SA_tst), normalize_0_1(X_SP_tst1), normalize_0_1(X_SP_tst2), normalize_0_1(X_SP_tst3)

## test data
Xt = np.concatenate([X_SA_tst, X_SP_tst], axis = 0)
yt = np.concatenate([np.zeros((len(X_SA_tst),1)), np.ones((len(X_SP_tst),1))], axis = 0).flatten()
## Dimension adjust
X_SA_tst, X_SP_tst1, X_SP_tst2, X_SP_tst3, Xt = np.expand_dims(X_SA_tst, axis = 3), np.expand_dims(X_SP_tst1, axis = 3), np.expand_dims(X_SP_tst2, axis = 3),\
		 np.expand_dims(X_SP_tst3, axis = 3), np.expand_dims(Xt, axis = 3)
print_red('Data Loaded !')

# batch_norm = True 
bn_training = False
scope = 'base'
x = tf.placeholder("float", shape=[None, 256, 256, 1])
is_training = tf.placeholder_with_default(False, (), 'is_training')

if version == 1:
	h1, h2, y = auto_encoder(x, nb_cnn = nb_cnn, bn = batch_norm, bn_training = is_training, filters = filters, kernel_size = [kernel_size, kernel_size], scope_name = scope)
elif version == 2:
	h1, h2, y = auto_encoder(x, nb_cnn = nb_cnn, bn = batch_norm, bn_training = is_training, filters = filters, kernel_size = [kernel_size, kernel_size], scope_name = scope)
sqr_err = tf.square(y - x)

# tf.keras.backend.clear_session()
# create a saver
vars_list = tf.global_variables(scope)
key_list = [v.name[:-2] for v in tf.global_variables(scope)]
key_direct = {}
for key, var in zip(key_list, vars_list):
	key_direct[key] = var
saver = tf.train.Saver(key_direct, max_to_keep=1)

# print out trainable parameters
for v in key_list:
	print_green(v)

# evaluate the model
with tf.Session() as sess:
	tf.global_variables_initializer().run(session=sess)
	saver.restore(sess, model_folder+'/best')
# 	saver.restore(sess, model_folder + '/model-1900')
# 	tf.reset_default_graph()
# 	imported_graph = tf.train.import_meta_graph(model_folder + '/model-4500.meta')
# 	imported_graph.restore(sess, model_folder + '/model-4500')
	# reconstructed images
	Yn = y.eval(session = sess, feed_dict = {x: X_SA_tst, is_training: False}); Ya = y.eval(session = sess, feed_dict = {x: X_SP_tst, is_training: False})
	
	y_recon = np.concatenate([Yn, Ya], axis = 0)
	# reconstruction errors-based detection
	norm_err_map_list = []
	norm_err_n_list = []
	for i in range(X_SA_tst.shape[0]):
		norm_image = X_SA_tst[i,:].reshape((1,256,256,1))
		norm_err = sqr_err.eval(session = sess, feed_dict = {x: norm_image, is_training: False})
		norm_err_n = (norm_err - np.min(norm_err))/(np.max(norm_err)-np.min(norm_err))
		norm_err_map_list.append(norm_err); norm_err_n_list.append(norm_err_n)
	norm_err_map_arr = np.concatenate(norm_err_map_list, axis = 0); norm_err_n_arr = np.concatenate(norm_err_n_list, axis = 0)
	anomaly_err_map_list = []
	anomaly_err_n_list = []
	for i in range(X_SP_tst.shape[0]):
		anomaly_image = X_SP_tst[i,:].reshape((1,256,256,1))
		anomaly_err = sqr_err.eval(session = sess, feed_dict = {x: anomaly_image, is_training: False})
		anomaly_err_n = (anomaly_err - np.min(anomaly_err))/(np.max(anomaly_err)-np.min(anomaly_err))
		anomaly_err_map_list.append(anomaly_err); anomaly_err_n_list.append(anomaly_err_n)
	anomaly_err_map_arr = np.concatenate(anomaly_err_map_list, axis = 0); anomaly_err_n_arr = np.concatenate(anomaly_err_n_list, axis = 0)
	norm_err_map = sqr_err.eval(session = sess, feed_dict = {x: X_SA_tst, is_training: False}); anomaly_err_map = sqr_err.eval(session = sess, feed_dict = {x: X_SP_tst, is_training: False})
	print('Difference: SA {0:.4f} SP {1:.4f}'.format(np.sum(np.abs(norm_err_map_arr - norm_err_map)), np.sum(np.abs(anomaly_err_map_arr - anomaly_err_map))))
	recon_err_map = np.concatenate([norm_err_map, anomaly_err_map], axis = 0)
	recon_err_n_map = np.concatenate([norm_err_n_arr, anomaly_err_n_arr])
	recon_err_map1 = np.concatenate([norm_err_map_arr, anomaly_err_map_arr], axis = 0)
	recon_errs = np.apply_over_axes(np.mean, recon_err_map, [1,2,3]).flatten(); AE_auc = roc_auc_score(yt, recon_errs)
	recon_errs1 = np.apply_over_axes(np.mean, recon_err_map1, [1,2,3]).flatten(); AE_auc1 = roc_auc_score(yt, recon_errs1)
	recon_errs_n = np.apply_over_axes(np.mean, recon_err_n_map, [1,2,3]).flatten(); AE_auc_n = roc_auc_score(yt, recon_errs_n)
	# pixel mean-based detection
	img_means = np.squeeze(np.apply_over_axes(np.mean, Xt, axes = [1,2,3])); MP_auc = roc_auc_score(yt, img_means)
	print_yellow('AUC: AE {0:.4f} AE(compare) {1:.4f} AE(normalized) {2:.4f} MP: {3:.4f}'.format(AE_auc, AE_auc1, AE_auc_n, MP_auc))
	print(model_name)
	np.savetxt(os.path.join(model_folder,'AE_stat.txt'), recon_errs)
	np.savetxt(os.path.join(model_folder,'MP_stat.txt'), img_means)
	np.savetxt(os.path.join(model_folder,'best_auc.txt'),[AE_auc, MP_auc])
	hist_file = os.path.join(model_folder,'hist-{}.png'.format(model_name))
	plot_hist(hist_file, recon_errs[:int(len(recon_errs)/2)], recon_errs[int(len(recon_errs)/2):])
	plot_hist_pixels(model_folder+'/hist_mean_pixel.png'.format(model_name), img_means[:int(len(img_means)/2)], img_means[int(len(img_means)/2):])
	saver.save(sess, model_folder +'/best')
	save_recon_images(model_folder+'/recon-{}.png'.format(model_name), Xt, y_recon, recon_err_map, fig_size = [11,5])
	