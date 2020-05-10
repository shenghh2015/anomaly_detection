import tensorflow as tf
import numpy as np
import os

def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)

def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)

def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding="SAME")

def lrelu(x, alpha=0.2):
  return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding="SAME")

l2_regularizer = tf.contrib.layers.l2_regularizer(1e-5)
def _conv_bn_lrelu_pool(x, pool = False, bn = True):
	_conv = tf.layers.conv2d(x, filters = 32, kernel_size = [5,5], strides=(1, 1), padding='same',
			kernel_initializer= 'truncated_normal', kernel_regularizer=l2_regularizer)
	if bn:
		_bn = tf.layers.batch_normalization(_conv, training = True)
	else:
		_bn = _conv
	_lrelu = tf.nn.leaky_relu(_bn)
	if pool:
		_out = max_pool_2x2(_lrelu)
	else:
		_out = _lrelu
	return _out

def conv_block(x, nb_cnn = 4, bn = False, scope_name = 'base'):
	with tf.variable_scope(scope_name):
		h = _conv_bn_lrelu_pool(x, pool = False, bn = bn)
		for i in range(1, nb_cnn):
			if i%2 == 1:
				pool = True
			else:
				pool = False
			h = _conv_bn_lrelu_pool(h, pool = pool, bn = bn)
	return h

# x = tf.placeholder("float", shape=[None, 109,109, 1])
# h = conv_block(x, nb_cnn = 4, bn = True, scope_name = 'base')

def dense_block(x, fc_layers = [128, 1], bn = False, scope_name = 'base'):
# 	shape = x.shape.as_list()[1:]
	with tf.variable_scope(scope_name):
		flat = tf.layers.flatten(x)
		h1 = tf.layers.dense(flat, fc_layers[0], kernel_regularizer=l2_regularizer)
		if bn:
			h1 = tf.layers.batch_normalization(h1, training = True)
		h1 = tf.nn.leaky_relu(h1)
		h2 = tf.layers.dense(h1, fc_layers[1], kernel_regularizer = l2_regularizer)
	return h1, h2

### create network
def conv_classifier(x, nb_cnn = 4, fc_layers = [128,1],  bn = False, scope_name = 'base', reuse = False):
	with tf.variable_scope(scope_name, reuse = reuse):
		conv_net = conv_block(x, nb_cnn = nb_cnn, bn = bn, scope_name = 'conv')
		h, pred_logit = dense_block(conv_net, fc_layers = fc_layers, bn = bn, scope_name = 'classifier')

	return conv_net, h, pred_logit

def conv_classifier2(x, nb_cnn = 4, fc_layers = [128,1],  bn = False, scope_name = 'base', reuse = False):
	with tf.variable_scope(scope_name, reuse = reuse):
		conv_net = conv_block(x, nb_cnn = nb_cnn, bn = bn, scope_name = 'conv')
		h, pred_logit = dense_block(conv_net, fc_layers = fc_layers, bn = True, scope_name = 'classifier')

	return conv_net, h, pred_logit

def discriminator(x, nb_cnn = 2, fc_layers = [128, 1], bn = True, reuse = False):
	with tf.variable_scope('discriminator', reuse = reuse):
		if nb_cnn > 0:
			h = conv_block(x, nb_cnn = nb_cnn, bn = bn, scope_name = 'cov')
			_, pred_logit = dense_block(h, fc_layers = fc_layers, bn = bn, scope_name = 'fc')
		else:
			_, pred_logit = dense_block(x, fc_layers = fc_layers, bn = bn, scope_name = 'fc')
	return pred_logit
