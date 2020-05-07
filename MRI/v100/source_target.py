import tensorflow as tf
import numpy as np
import argparse
import glob
import os
import scipy.io
import math
import time
from sklearn.metrics import roc_auc_score

from load_data import *
from model import *

def print_yellow(str):
	from termcolor import colored 
	print(colored(str, 'yellow'))

def print_red(str):
	from termcolor import colored 
	print(colored(str, 'red'))

def print_green(str):
	from termcolor import colored 
	print(colored(str, 'green'))

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
source = '/data/results/CLB'
target = '/data/results/FDA'
# source_model_name = 'noise-2.0-trn-100000-sig-0.035-bz-400-lr-5e-05-Adam-100.0k'
source_model_name = 'cnn-4-bn-True-noise-2.0-trn-100000-sig-0.035-bz-400-lr-5e-05-Adam-4.0k'
source_model_file = os.path.join(source, source_model_name,'source-best')
nb_train = 85000

# load target images
_, X_val, X_tst, _, y_val, y_tst = load_target(dataset = 'total', train = nb_train)
X_val, X_tst = (X_val-np.min(X_val))/(np.max(X_val)-np.min(X_val)), (X_tst-np.min(X_tst))/(np.max(X_tst)-np.min(X_tst))
X_val, X_tst = np.expand_dims(X_val, axis = 3), np.expand_dims(X_tst, axis = 3)
y_val, y_tst = y_val.reshape(-1,1), y_tst.reshape(-1,1)

if source_model_name.split('-')[0] == 'cnn':
	nb_cnn = int(source_model_name.split('-')[1])
else:
	nb_cnn = 4

if source_model_name.split('-')[2] == 'bn':
	bn = bool(source_model_name.split('-')[3])
else:
	bn = False

xs = tf.placeholder("float", shape=[None, 109,109, 1])
ys = tf.placeholder("float", shape=[None, 1])
conv_net_src, h_src, source_logit = conv_classifier(xs, nb_cnn = nb_cnn, fc_layers = [128,1],  bn = bn, scope_name = 'source')
source_vars_list = tf.trainable_variables('source')
source_key_list = [v.name[:-2].replace('source', 'base') for v in tf.trainable_variables('source')]
source_key_direct = {}
for key, var in zip(source_key_list, source_vars_list):
	source_key_direct[key] = var
source_saver = tf.train.Saver(source_key_direct, max_to_keep=1)

## model loading verification
with tf.Session() as sess:
	tf.global_variables_initializer().run(session=sess)
	source_saver.restore(sess, source_model_file)
	# source to target (target loading)
	# valid set
	val_target_logit = source_logit.eval(session=sess,feed_dict={xs:X_val})
	val_target_stat = np.exp(val_target_logit)
	val_target_AUC = roc_auc_score(y_val, val_target_stat)
	# test set
	test_target_logit = source_logit.eval(session=sess,feed_dict={xs:X_tst})
	test_target_stat = np.exp(test_target_logit)
	test_target_AUC = roc_auc_score(y_tst, test_target_stat)
	print_yellow('AUC: source-target-T {0:.4f} -V {1:.4f}'.format(test_target_AUC, val_target_AUC))
	# save result
	with open(source+'/'+source_model_name+'/source_to_target.txt', 'w+') as f:
		f.write('>>>> Source {} To Target {} >>>>'.format(os.path.basename(source), os.path.basename(target)))
		f.write(' -AUC: valid {0:.4f}, test {1:.4f}\n'.format(test_target_AUC, val_target_AUC))