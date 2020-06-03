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
import matplotlib

from load_data import load_MRI_anomaly, load_MRI_anomaly_test
from models2 import auto_encoder, auto_encoder3, auto_encoder4
from helper_function import normalize_0_1, print_yellow, print_red, print_green, print_block
from helper_function import plot_hist, plot_LOSS, plot_AUC, plot_hist_pixels, plot_hist_list
from helper_function import generate_folder, save_recon_images, save_recon_images_v2, save_recon_images_v3, save_recon_images_v4 

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

# model_name = 'AE1-MRI-cn-6-fr-32-ks-5-bn-False-lr-0.0001-stps-100000-bz-50-tr-65k-vl-400-test-1000-l-mse'
# model_name = 'AE1-MRI-cn-4-fr-32-ks-5-bn-False-lr-0.0001-stps-100000-bz-50-tr-65k-vl-400-test-1000-l-mse'
# model_name = 'AE4-MRI-cn-6-fr-32-ks-5-bn-False-lr-0.0001-stps-100000-bz-50-tr-65k-vl-400-test-1000-l-mae'
# model_name = 'AEL3-MRI-cn-4-fr-32-ks-5-bn-False-lr-0.0001-stps-100000-bz-50-tr-65k-vl-400-test-1000-l-mae'
model_name = 'AEL1-MRI-cn-4-fr-32-ks-5-bn-True-lr-5e-06-stps-100000-bz-50-tr-65k-vl-400-test-1000-l-mae-ano_w-0.05-3x'

splits = model_name.split('-')
if len(splits[0])<=2:
	version =1
else:
	version = int(splits[0][-1])
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
	elif splits[i] == 'l':
		loss = splits[i+1]


model_folder = os.path.join(output_folder, model_name)
dim = 256
## load data
print_red('Data loading ...')
Xn = load_MRI_anomaly_test('true')
Xa2, Xa3, Xa4 = load_MRI_anomaly_test('meas_2x'), load_MRI_anomaly_test('meas_3x'), load_MRI_anomaly_test('meas_4x')
Xam2, Xam4 = load_MRI_anomaly_test('null_mixed_2x'), load_MRI_anomaly_test('null_mixed_4x')
Xn, Xa2, Xa3, Xa4, Xam2, Xam4 = normalize_0_1(Xn), normalize_0_1(Xa2), normalize_0_1(Xa3), normalize_0_1(Xa4), normalize_0_1(Xam2), normalize_0_1(Xam4)

# create a graph
scope = 'base'
x = tf.placeholder("float", shape=[None, dim, dim, 1])
is_training = tf.placeholder_with_default(False, (), 'is_training')

if version == 1 or version == 2:
	h1, h2, y = auto_encoder(x, nb_cnn = nb_cnn, bn = batch_norm, bn_training = is_training, filters = filters, kernel_size = [kernel_size, kernel_size], scope_name = scope)
elif version == 3:
	h1, h2, y = auto_encoder3(x, nb_cnn = nb_cnn, bn = batch_norm, bn_training = is_training, filters = filters, kernel_size = [kernel_size, kernel_size], scope_name = scope)
elif version == 4:
	h1, h2, y = auto_encoder4(x, nb_cnn = nb_cnn, bn = batch_norm, bn_training = is_training, filters = filters, kernel_size = [kernel_size, kernel_size], scope_name = scope)

if loss == 'mse':
	err_map = tf.square(y - x)
elif loss == 'mae':
	err_map = tf.abs(y - x)

# tf.keras.backend.clear_session()
# create a saver
vars_list = tf.global_variables(scope)
key_list = [v.name[:-2] for v in tf.global_variables(scope)]
key_direct = {}
for key, var in zip(key_list, vars_list):
	key_direct[key] = var
saver = tf.train.Saver(key_direct, max_to_keep=1)

# print out trainable parameters
# for v in key_list:
# 	print_green(v)

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

# evaluate the reconstruction errs
result_folder = model_folder + '/evaluation_results'
generate_folder(result_folder)
with tf.Session() as sess:
	tf.global_variables_initializer().run(session=sess)
	saver.restore(sess, model_folder+'/model-80000')
	
	# reconstruction and reconstruction err map
	Yn, map_n = evaluate(sess, y, x, is_training, err_map, Xn.reshape(-1,dim,dim,1), batch_size = 100)
	Ya2, map_a2 = evaluate(sess, y, x, is_training, err_map, Xa2.reshape(-1,dim,dim,1), batch_size = 100)
	Ya3, map_a3 = evaluate(sess, y, x, is_training, err_map, Xa3.reshape(-1,dim,dim,1), batch_size = 100)
	Ya4, map_a4 = evaluate(sess, y, x, is_training, err_map, Xa4.reshape(-1,dim,dim,1), batch_size = 100)
	Yam2, map_am2 = evaluate(sess, y, x, is_training, err_map, Xam2.reshape(-1,dim,dim,1), batch_size = 100)
	Yam4, map_am4 = evaluate(sess, y, x, is_training, err_map, Xam4.reshape(-1,dim,dim,1), batch_size = 100)
	
	# recon err
	err_n = np.apply_over_axes(np.mean, map_n, [1,2,3]).flatten()
	err_a2 = np.apply_over_axes(np.mean, map_a2, [1,2,3]).flatten()
	err_a3 = np.apply_over_axes(np.mean, map_a3, [1,2,3]).flatten()
	err_a4 = np.apply_over_axes(np.mean, map_a4, [1,2,3]).flatten()
	err_am2 = np.apply_over_axes(np.mean, map_am2, [1,2,3]).flatten()
	err_am4 = np.apply_over_axes(np.mean, map_am4, [1,2,3]).flatten()
	
	total_errs = np.concatenate([err_n, err_a2, err_a3, err_a4, err_am2, err_am4])

	# recon images and err map
	save_recon_images_v3(result_folder+'/rec_{}-true.png'.format(model_name), Xn, Yn, map_n, fig_size = [11,20])
	save_recon_images_v3(result_folder+'/rec_{}-meas_2x.png'.format(model_name), Xa2, Ya2, map_a2, fig_size = [11,20])
	save_recon_images_v3(result_folder+'/rec_{}-meas-3x.png'.format(model_name), Xa3, Ya3, map_a3, fig_size = [11,20])
	save_recon_images_v3(result_folder+'/rec_{}-meas-4x.png'.format(model_name), Xa4, Ya4, map_a4, fig_size = [11,20])	
	save_recon_images_v3(result_folder+'/rec_{}-null-mix-2x.png'.format(model_name), Xam2, Yam2, map_am2, fig_size = [11,20])	
	save_recon_images_v3(result_folder+'/rec_{}-null-mix-4x.png'.format(model_name), Xam4, Yam4, map_am4, fig_size = [11,20])	
		
	# plot histogram of errs
	min_value, max_value = np.min(total_errs), np.max(total_errs)
	plot_hist_list(result_folder+'/hist-{}-{}.png'.format(model_name, 'meas_4x'), [err_n, err_a4], ['f_true', 'f_meas_4x'], ['g', 'c'], [min_value, max_value])
	plot_hist_list(result_folder+'/hist-{}-{}.png'.format(model_name, 'meas_3x'), [err_n, err_a3], ['f_true', 'f_meas_3x'], ['g', 'r'], [min_value, max_value])
	plot_hist_list(result_folder+'/hist-{}-{}.png'.format(model_name, 'meas_2x'), [err_n, err_a2], ['f_true', 'f_meas_2x'], ['g', 'b'], [min_value, max_value])
	plot_hist_list(result_folder+'/hist-{}-{}.png'.format(model_name, 'null_mix_4x'), [err_n, err_am4], ['f_true','f_null_mix_4x'], ['g', 'y'], [min_value, max_value])
	plot_hist_list(result_folder+'/hist-{}-{}.png'.format(model_name, 'null_mix_2x'), [err_n, err_am2], ['f_true','f_null_mix_2x'], ['g', 'k'], [min_value, max_value])
