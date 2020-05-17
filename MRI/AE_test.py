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
from helper_function import plot_hist, plot_LOSS, plot_AUC, plot_hist_pixels, plot_hist_list
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
# model_name = 'AE1-MRI-cn-4-fr-32-ks-5-bn-False-lr-0.0001-stps-100000-bz-50-tr-65k-vl-400-test-1000-l-mse'

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
_, _, X_SA_tst, X_SP_tst1 = load_MRI_anomaly(docker = docker, train = train, val = val, normal = test, anomaly = test, version = 1)
_, _, _, X_SP_tst2 = load_MRI_anomaly(docker = docker, train = train, val = val, normal = test, anomaly = test, version = 2)
_, _, _, X_SP_tst3 = load_MRI_anomaly(docker = docker, train = train, val = val, normal = test, anomaly = test, version = 3)
X_SA_tst, X_SP_tst1, X_SP_tst2, X_SP_tst3 = normalize_0_1(X_SA_tst), normalize_0_1(X_SP_tst1), normalize_0_1(X_SP_tst2), normalize_0_1(X_SP_tst3)

## test data
Xt = np.concatenate([X_SA_tst, X_SP_tst1, X_SP_tst2, X_SP_tst3], axis = 0)
#yt = np.concatenate([np.zeros((len(X_SA_tst),1)), np.ones((len(X_SP_tst1),1))], axis = 0).flatten()
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
err_map = tf.square(y - x)

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

def evaluate(sess, y, x, is_training, err_map, X_tst, batch_size = 100):
	y_list, err_map_list = [], []
	i = 0
	while batch_size*i < X_tst.shape[0]:
		batch_x = X_tst[batch_size*i: min(batch_size*(i+1), X_tst.shape[0]),:]
		y_recon = y.eval(session = sess, feed_dict = {x:batch_x,is_training: False})
		y_list.append(y_recon)
		err_map_list.append(err_map.eval(session = sess, feed_dict = {x:batch_x,is_training: False}))
		i = i +1
	y_arr, err_map_arr = np.concatenate(y_list, axis = 0), np.concatenate(err_map_list, axis = 0)
	return y_arr, err_map_arr

# evaluate the model
with tf.Session() as sess:
	tf.global_variables_initializer().run(session=sess)
	saver.restore(sess, model_folder+'/best')
	# norm
	Yn, norm_err_map = evaluate(sess, y, x, is_training, err_map, X_SA_tst, batch_size = 100)
	Ya1, anom_err_map1 = evaluate(sess, y, x, is_training, err_map, X_SP_tst1, batch_size = 100)
	Ya2, anom_err_map2 = evaluate(sess, y, x, is_training, err_map, X_SP_tst2, batch_size = 100)
	Ya3, anom_err_map3 = evaluate(sess, y, x, is_training, err_map, X_SP_tst3, batch_size = 100)
	norm_recon_errs = np.apply_over_axes(np.mean, norm_err_map, [1,2,3]).flatten()
	anom_recon_errs1 = np.apply_over_axes(np.mean, anom_err_map1, [1,2,3]).flatten()
	anom_recon_errs2 = np.apply_over_axes(np.mean, anom_err_map2, [1,2,3]).flatten()
	anom_recon_errs3 = np.apply_over_axes(np.mean, anom_err_map3, [1,2,3]).flatten()
	
	imgs = np.concatenate([X_SA_tst, X_SP_tst1, X_SP_tst2, X_SP_tst3], axis = 0)
	recons = np.concatenate([Yn, Ya1, Ya2, Ya3], axis = 0)
	err_maps = np.concatenate([norm_err_map, anom_err_map1, anom_err_map2, anom_err_map3], axis = 0)
	recon_errs = np.concatenate([norm_recon_errs, anom_recon_errs1, anom_recon_errs2, anom_recon_errs3], axis = 0)
# 	recon_errs = np.apply_over_axes(np.mean, err_maps, [1,2,3]).flatten()
# 	print_yellow('AUC: AE {0:.4f} AE(compare) {1:.4f} AE(normalized) {2:.4f} MP: {3:.4f}'.format(AE_auc, AE_auc1, AE_auc_n, MP_auc))
	print(model_name)
	result_folder = model_folder + '/detection_results'
	generate_folder(result_folder)
	np.savetxt(os.path.join(result_folder,'norm_stat.txt'), norm_recon_errs)
	np.savetxt(os.path.join(result_folder,'anom_stat1.txt'), anom_recon_errs1)
	np.savetxt(os.path.join(result_folder,'anom_stat2.txt'), anom_recon_errs2)
	np.savetxt(os.path.join(result_folder,'anom_stat3.txt'), anom_recon_errs3)
	## plot err histogram and recon images
	idx1, idx2, idx3 = int(len(recon_errs)/4), int(len(recon_errs)/2), int(len(recon_errs)*3/4)
	err_stat_list = [recon_errs[:idx1], recon_errs[idx1:idx2], recon_errs[idx2:idx3], recon_errs[idx3:]]
	max_value, min_value = np.min(recon_errs), np.max(recon_errs)
	print_green('Length: norm {} anom1 {} anom2 {} anom3 {}'.format(len(recon_errs[:idx1]), len(recon_errs[idx1:idx2]), len(recon_errs[idx2:idx3]), len(recon_errs[idx3:])))
	plot_hist_list(result_folder+'/hist-{}.png'.format(model_name), err_stat_list, ['Norm', 'Anomaly1', 'Anomaly2', 'Anomaly3'], ['g', 'r', 'b', 'y'], [max_value, min_value])
	plot_hist_list(result_folder+'/hist1-{}.png'.format(model_name), [recon_errs[:idx1], recon_errs[idx1:idx2]], ['Norm', 'Anomaly1'], ['g', 'r'], [max_value, min_value])
	plot_hist_list(result_folder+'/hist2-{}.png'.format(model_name), [recon_errs[:idx1], recon_errs[idx2:idx3]], ['Norm', 'Anomaly2'], ['g', 'b'], [max_value, min_value])
	plot_hist_list(result_folder+'/hist3-{}.png'.format(model_name), [recon_errs[:idx1], recon_errs[idx3:]], ['Norm','Anomaly3'], ['g', 'y'], [max_value, min_value])
	save_recon_images_v2(result_folder+'/recon-{}.png'.format(model_name), Xt, recons, err_maps, fig_size = [11,10])
	