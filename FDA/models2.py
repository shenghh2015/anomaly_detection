import tensorflow as tf
import numpy as np
import os

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding="SAME")

def bn_layer(x, scope, is_training, epsilon=0.001, decay=0.99, reuse=None):
    """
    Performs a batch normalization layer

    Args:
        x: input tensor
        scope: scope name
        is_training: python boolean value
        epsilon: the variance epsilon - a small float number to avoid dividing by 0
        decay: the moving average decay

    Returns:
        The ops of a batch normalization layer
    """
    with tf.variable_scope(scope, reuse=reuse):
        shape = x.get_shape().as_list()
        # gamma: a trainable scale factor
        gamma = tf.get_variable("gamma", shape[-1], initializer=tf.constant_initializer(1.0), trainable=True)
        # beta: a trainable shift value
        beta = tf.get_variable("beta", shape[-1], initializer=tf.constant_initializer(0.0), trainable=True)
        moving_avg = tf.get_variable("moving_avg", shape[-1], initializer=tf.constant_initializer(0.0), trainable=False)
        moving_var = tf.get_variable("moving_var", shape[-1], initializer=tf.constant_initializer(1.0), trainable=False)
        if is_training:
            # tf.nn.moments == Calculate the mean and the variance of the tensor x
            avg, var = tf.nn.moments(x, np.arange(len(shape)-1), keep_dims=True)
            avg=tf.reshape(avg, [avg.shape.as_list()[-1]])
            var=tf.reshape(var, [var.shape.as_list()[-1]])
            #update_moving_avg = moving_averages.assign_moving_average(moving_avg, avg, decay)
            update_moving_avg=tf.assign(moving_avg, moving_avg*decay+avg*(1-decay))
            #update_moving_var = moving_averages.assign_moving_average(moving_var, var, decay)
            update_moving_var=tf.assign(moving_var, moving_var*decay+var*(1-decay))
            control_inputs = [update_moving_avg, update_moving_var]
        else:
            avg = moving_avg
            var = moving_var
            control_inputs = []
        with tf.control_dependencies(control_inputs):
            output = tf.nn.batch_normalization(x, avg, var, offset=beta, scale=gamma, variance_epsilon=epsilon)

    return output


def bn_layer_top(x, scope, is_training, epsilon=0.001, decay=0.99):
    """
    Returns a batch normalization layer that automatically switch between train and test phases based on the 
    tensor is_training

    Args:
        x: input tensor
        scope: scope name
        is_training: boolean tensor or variable
        epsilon: epsilon parameter - see batch_norm_layer
        decay: epsilon parameter - see batch_norm_layer

    Returns:
        The correct batch normalization layer based on the value of is_training
    """
    #assert isinstance(is_training, (ops.Tensor, variables.Variable)) and is_training.dtype == tf.bool

    return tf.cond(
        is_training,
        lambda: bn_layer(x=x, scope=scope, epsilon=epsilon, decay=decay, is_training=True, reuse=None),
        lambda: bn_layer(x=x, scope=scope, epsilon=epsilon, decay=decay, is_training=False, reuse=True),
    )

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding="SAME")

l2_regularizer = tf.contrib.layers.l2_regularizer(1e-5)
def _conv_bn_lrelu_pool(x, filters = 32, kernel_size = [5,5], pool = False, bn = True, bn_training = True, scope = 'conv'):
	_conv = tf.layers.conv2d(x, filters = filters, kernel_size = kernel_size, strides=(1, 1), padding='same',
			kernel_initializer= 'truncated_normal', kernel_regularizer=l2_regularizer)
	if bn:
# 		_bn = tf.layers.batch_normalization(_conv, training = bn_training)
		_bn = bn_layer_top(_conv, scope, is_training = bn_training, epsilon=0.001, decay=0.99)
	else:
		_bn = _conv
	_lrelu = tf.nn.leaky_relu(_bn)
	if pool:
		_out = max_pool_2x2(_lrelu)
	else:
		_out = _lrelu
	return _out

def conv_block(x, nb_cnn = 4, bn = False, bn_training = True, filters = 32, kernel_size = [5,5], scope_name = 'encoder'):
	with tf.variable_scope(scope_name):
		h = _conv_bn_lrelu_pool(x, filters = filters, kernel_size = kernel_size, pool = False, bn = bn, bn_training = bn_training, scope = 'conv0')
		for i in range(1, nb_cnn):
			if i%2 == 1:
				pool = True
			else:
				pool = False
			h = _conv_bn_lrelu_pool(h, filters = filters, kernel_size = kernel_size, pool = pool, bn = bn, bn_training = bn_training, scope = 'conv{}'.format(i))
	return h

def _up_conv_bn_lrelu(x, filters = 32, kernel_size = [5,5], up = False, bn = True, bn_training = True, scope = 'conv'):
	if up:
		x = tf.keras.layers.UpSampling2D()(x)
	_conv = _conv_bn_lrelu_pool(x, filters = filters, kernel_size = kernel_size, pool = False, bn = bn, bn_training = bn_training, scope = scope)
	return _conv

def up_conv_block(x, nb_cnn = 4, bn = False, bn_training = True, filters = 32, kernel_size = [5,5], scope_name = 'decoder'):
	with tf.variable_scope(scope_name):
		h = _up_conv_bn_lrelu(x, filters = filters, kernel_size = kernel_size, up = True, bn = bn, bn_training = bn_training, scope = 'conv0')
		for i in range(1, nb_cnn):
			if i%2 == 1:
				up = False
			else:
				up = True
			h = _up_conv_bn_lrelu(h, filters = filters, kernel_size = kernel_size, up = up, bn = bn, bn_training = bn_training, scope = 'conv{}'.format(i))
	return h

### create network
def auto_encoder(x, nb_cnn = 4, bn = False, bn_training = True, filters = 32, kernel_size = [5,5], scope_name = 'base', reuse = False):
	with tf.variable_scope(scope_name, reuse = reuse):
		h1 = conv_block(x, nb_cnn = nb_cnn, bn = bn, bn_training = bn_training, filters = filters, kernel_size = kernel_size, scope_name = 'encoder')
		h2 = up_conv_block(h1, nb_cnn = nb_cnn, bn = bn, bn_training = bn_training, filters = filters, kernel_size = kernel_size, scope_name = 'decoder')
		y = tf.layers.conv2d(h2, filters = 1, kernel_size = kernel_size, strides=(1, 1), padding='same',
					kernel_initializer= 'truncated_normal', kernel_regularizer=l2_regularizer)
		y = tf.nn.relu(y)
	return h1, h2, y