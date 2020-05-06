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

from load_data import *
from model import *

## load dataset
X_SA_trn, X_SA_val, X_SA_tst, X_SP_tst = load_anomaly_data(dataset = 'dense', train = 7100, valid = 400, test = 400)

def lrelu(x, alpha=0.2):
  return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

def autoencoder(input_shape=[None, 109, 109, 1], n_filters=[1, 32, 32, 32], filter_sizes=[5, 5, 5, 5]):
	input_shape=[None, 109, 109]
	n_filters=[1, 32, 32, 32]
	filter_sizes=[5, 5, 5, 5]

	# input to the network
	x = tf.placeholder(tf.float32, input_shape, name='x')
	x_tensor = x
	current_input = x_tensor
# 
# 	# ensure 2-d is converted to square tensor.
# 	if len(x.get_shape()) == 2:
# 		x_dim = np.sqrt(x.get_shape().as_list()[1])
# 		if x_dim != int(x_dim):
# 			raise ValueError('Unsupported input dimensions')
# 		x_dim = int(x_dim)
# 		x_tensor = tf.reshape(x, [-1, x_dim, x_dim, n_filters[0]])
# 	elif len(x.get_shape()) == 4:
# 		x_tensor = x
# 	else:
# 		raise ValueError('Unsupported input dimensions')
# 	current_input = x_tensor

	# %%
	# Build the encoder
	encoder = []
	shapes = []
	for layer_i, n_output in enumerate(n_filters[1:]):
		print(layer_i, n_output)
		n_input = current_input.get_shape().as_list()[3]
		shapes.append(current_input.get_shape().as_list())
		W = tf.Variable(tf.random_uniform([filter_sizes[layer_i],filter_sizes[layer_i], n_input, n_output], -1.0 / math.sqrt(n_input), 1.0 / math.sqrt(n_input)))
		b = tf.Variable(tf.zeros([n_output]))
		encoder.append(W)
		output = lrelu(tf.add(tf.nn.conv2d(current_input, W, strides=[1, 2, 2, 1], padding='SAME'), b))
		current_input = output

	# %%
	# store the latent representation
	z = current_input
	encoder.reverse()
	shapes.reverse()

	# %%
	# Build the decoder using the same weights
	for layer_i, shape in enumerate(shapes):
		W = encoder[layer_i]
		b = tf.Variable(tf.zeros([W.get_shape().as_list()[2]]))
		output = lrelu(tf.add(
			tf.nn.conv2d_transpose(
				current_input, W,
				tf.stack([tf.shape(x)[0], shape[1], shape[2], shape[3]]),
				strides=[1, 2, 2, 1], padding='SAME'), b))
		current_input = output

	# %%
	# now have the reconstruction through the network
	y = current_input
	# cost function measures pixel-wise difference
	cost = tf.reduce_sum(tf.square(y - x_tensor))

	# %%
	return {'x': x, 'z': z, 'y': y, 'cost': cost}