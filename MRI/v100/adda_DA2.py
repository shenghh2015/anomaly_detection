import tensorflow as tf
import tensorflow.math as tm

import numpy as np
import os
import glob
from natsort import natsorted
from termcolor import colored 
import argparse
from sklearn.metrics import roc_auc_score
import scipy.io
import time

from load_data import *
from model import *

from functools import partial

def str2bool(value):
    return value.lower() == 'true'

def plot_LOSS(file_name, train_loss_list, val_loss_list, test_loss_list):
	import matplotlib.pyplot as plt
	from matplotlib.backends.backend_agg import FigureCanvasAgg
	from matplotlib.figure import Figure
	fig_size = (8,6)
	fig = Figure(figsize=fig_size)
	ax = fig.add_subplot(111)
	ax.plot(train_loss_list)
	ax.plot(val_loss_list)
	ax.plot(test_loss_list)
	title = os.path.basename(os.path.dirname(file_name))
	ax.set_title(title)
	ax.set_xlabel('Iterations')
	ax.set_ylabel('Loss')
	ax.legend(['D','S','T'])
	ax.set_xlim([0,len(train_loss_list)])
	canvas = FigureCanvasAgg(fig)
	canvas.print_figure(file_name, dpi=100)

def plot_AUCs(file_name, train_list, val_list, test_list):
	import matplotlib.pyplot as plt
	from matplotlib.backends.backend_agg import FigureCanvasAgg
	from matplotlib.figure import Figure
	fig_size = (8,6)
	fig = Figure(figsize=fig_size)
	file_name = file_name
	ax = fig.add_subplot(111)
	ax.plot(train_list)
	ax.plot(val_list)
	ax.plot(test_list)
	title = os.path.basename(os.path.dirname(file_name))
	ax.set_title(title)
	ax.set_xlabel('Iterations')
	ax.set_ylabel('AUC')
	ax.legend(['Train','Valid','Test'])
	ax.set_xlim([0,len(train_list)])
	canvas = FigureCanvasAgg(fig)
	canvas.print_figure(file_name, dpi=100)

def plot_AUCs_DomACC(file_name, train_list, val_list, test_list, dom_acc_list):
	import matplotlib.pyplot as plt
	from matplotlib.backends.backend_agg import FigureCanvasAgg
	from matplotlib.figure import Figure
	fig_size = (8,6)
	fig = Figure(figsize=fig_size)
	file_name = file_name
	ax = fig.add_subplot(111)
	ax.plot(train_list)
	ax.plot(val_list)
	ax.plot(test_list)
	ax.plot(dom_acc_list)
	title = os.path.basename(os.path.dirname(file_name))
	ax.set_title(title)
	ax.set_xlabel('Iterations')
	ax.set_ylabel('AUC/ACC')
	ax.legend(['AUC:Train','AUC:Valid','AUC:Test','ACC:Dom'])
	ax.set_xlim([0,len(train_list)])
	canvas = FigureCanvasAgg(fig)
	canvas.print_figure(file_name, dpi=100)

# generate the folder
def generate_folder(folder):
    import os
    if not os.path.exists(folder):
        os.makedirs(folder)

def compute_pairwise_distances(x, y):
    if not len(x.get_shape()) == len(y.get_shape()) == 2:
        raise ValueError('Both inputs should be matrices.')
    if x.get_shape().as_list()[1] != y.get_shape().as_list()[1]:
        raise ValueError('The number of features should be the same.')

    norm = lambda x: tf.reduce_sum(tf.square(x), 1)
    return tf.transpose(norm(tf.expand_dims(x, 2) - tf.transpose(y)))

def gaussian_kernel_matrix(x, y, sigmas):
    beta = 1. / (2. * (tf.expand_dims(sigmas, 1)))
    dist = compute_pairwise_distances(x, y)
    s = tf.matmul(beta, tf.reshape(dist, (1, -1)))
    return tf.reshape(tf.reduce_sum(tf.exp(-s), 0), tf.shape(dist))


def maximum_mean_discrepancy(x, y, kernel=gaussian_kernel_matrix):
    cost = tf.reduce_mean(kernel(x, x))
    cost += tf.reduce_mean(kernel(y, y))
    cost -= 2 * tf.reduce_mean(kernel(x, y))
    cost = tf.where(cost > 0, cost, 0, name='value')
    return cost

def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)

# plot and save the file
def plot_loss(model_name, loss, val_loss, file_name):
	generate_folder(model_name)
	f_out = file_name
	from matplotlib.backends.backend_agg import FigureCanvasAgg
	from matplotlib.figure import Figure
	start_idx = 0
	if len(loss)>start_idx:
		title = os.path.basename(os.path.dirname(file_name))
		fig = Figure(figsize=(8,6))
		ax = fig.add_subplot(1,1,1)
		ax.plot(loss[start_idx:],'b-',linewidth=1.3)
		ax.plot(val_loss[start_idx:],'r-',linewidth=1.3)
		ax.set_title(title)
		ax.set_ylabel('Loss')
		ax.set_xlabel('batches')
		ax.legend(['D-loss', 'G-loss'], loc='upper left')  
		canvas = FigureCanvasAgg(fig)
		canvas.print_figure(f_out, dpi=80)

def plot_auc_iterations(target_auc_list, val_auc_list, target_file_name):
	import matplotlib.pyplot as plt
	from matplotlib.backends.backend_agg import FigureCanvasAgg
	from matplotlib.figure import Figure
	fig_size = (8,6)
	fig = Figure(figsize=fig_size)
	file_name = target_file_name
	ax = fig.add_subplot(111)
	ax.plot(target_auc_list)
	ax.plot(val_auc_list)
	title = os.path.basename(os.path.dirname(file_name))
	ax.set_title(title)
	ax.set_xlabel('Iterations')
	ax.set_ylabel('AUC')
	ax.legend(['Test','Val'])
	ax.set_xlim([0,len(target_auc_list)])
	canvas = FigureCanvasAgg(fig)
	canvas.print_figure(file_name, dpi=100)

def plot_auc_dom_acc_iterations(target_auc_list, val_auc_list, dom_acc_list, target_file_name):
	import matplotlib.pyplot as plt
	from matplotlib.backends.backend_agg import FigureCanvasAgg
	from matplotlib.figure import Figure
	fig_size = (8,6)
	fig = Figure(figsize=fig_size)
	file_name = target_file_name
	ax = fig.add_subplot(111)
	ax.plot(target_auc_list)
	ax.plot(val_auc_list)
	ax.plot(dom_acc_list)
	title = os.path.basename(os.path.dirname(file_name))
	ax.set_title(title)
	ax.set_xlabel('Iterations')
	ax.set_ylabel('AUC/ACC')
	ax.legend(['AUC:Test','AUC:Val', 'ACC:Dom'])
	ax.set_xlim([0,len(target_auc_list)])
	canvas = FigureCanvasAgg(fig)
	canvas.print_figure(file_name, dpi=100)

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
	print_red(symbol*nb_sybl)

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default = 0)
parser.add_argument("--docker", type = str2bool, default = True)
parser.add_argument("--shared", type = str2bool, default = True)
parser.add_argument("--dis_cnn", type = int, default = 0)
parser.add_argument("--g_cnn", type=int, default = 4)
parser.add_argument("--g_lr", type=float, default = 1e-5)
parser.add_argument("--d_lr", type=float, default = 1e-5)
parser.add_argument("--lsmooth", type=str2bool, default = False)
parser.add_argument("--dis_fc", type=int, default = 128)
parser.add_argument("--dis_bn", type=str2bool, default = True)
parser.add_argument("--nD", type = int, default = 1)
parser.add_argument("--nG", type = int, default = 1)
parser.add_argument("--acc_up", type = float, default = 0.8)
parser.add_argument("--acc_down", type = float, default = 0.3)
parser.add_argument("--lr", type = float, default = 1e-5)
parser.add_argument("--iters", type = int, default = 1000)
parser.add_argument("--bz", type = int, default = 400)
parser.add_argument("--dis_param", type = float, default = 1.0)
parser.add_argument("--trg_clf_param", type = float, default = 1.0)
parser.add_argument("--src_clf_param", type = float, default = 1.0)
parser.add_argument("--source_scratch", type = str2bool, default = True)
parser.add_argument("--nb_trg_labels", type = int, default = 0)
parser.add_argument("--fc_layer", type = int, default = 128)
parser.add_argument("--den_bn", type = str2bool, default = False)
parser.add_argument("--clf_v", type = int, default = 1)
parser.add_argument("--dataset", type = str, default = 'dense')

args = parser.parse_args()
print(args)

gpu_num = args.gpu
docker = args.docker
shared = args.shared
batch_size = args.bz
nb_steps = args.iters
dis_param = args.dis_param
dis_cnn = args.dis_cnn
dis_fc = args.dis_fc
dis_bn = args.dis_bn
nd_steps = args.nD
ng_steps = args.nG
acc_up = args.acc_up
acc_down = args.acc_down
lr = args.lr
nb_trg_labels = args.nb_trg_labels
source_scratch = args.source_scratch
fc_layer = args.fc_layer
den_bn = args.den_bn
dis_bn = args.dis_bn
trg_clf_param = args.trg_clf_param
src_clf_param = args.src_clf_param
clf_v = args.clf_v
dataset = args.dataset
g_lr = args.g_lr
d_lr = args.d_lr
lsmooth = args.lsmooth

g_cnn = args.g_cnn

if False:
	gpu_num = 1
	lr = 1e-5
	batch_size = 400
	nb_steps = 1000
	dis_param = 1.0
	nb_trg_labels = 0
	source_scratch = True
	docker = True
	shared = True
	fc_layer = 128
	den_bn = False

os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)

if docker:
	output_folder ='/data/results'
else:
	output_folder = 'data'

print(output_folder)
# hyper-parameters
noise = 2.0
sig_rate = 0.035
# source_model_name = 'cnn-4-bn-True-noise-2.0-trn-100000-sig-0.035-bz-400-lr-5e-05-Adam-4.0k'
source_model_name = 'cnn-4-bn-False-noise-2.0-trn-100000-sig-0.035-bz-400-lr-5e-05-Adam-100.0k'
# load source data
# source = '/data/results/CLB'
# target = '/data/results/FDA'
source = os.path.join(output_folder,'CLB')
target = os.path.join(output_folder,'FDA')
source_model_file = os.path.join(source, source_model_name, 'source-best')

# load source data
nb_source = 100000
Xs_trn, Xs_val, Xs_tst, ys_trn, ys_val, ys_tst = load_source(train = nb_source, sig_rate = sig_rate)
Xs_trn, Xs_val, Xs_tst = np.random.RandomState(2).normal(Xs_trn, noise), np.random.RandomState(0).normal(Xs_val, noise), np.random.RandomState(1).normal(Xs_tst, noise)
Xs_trn, Xs_val, Xs_tst = (Xs_trn-np.min(Xs_trn))/(np.max(Xs_trn)-np.min(Xs_trn)), (Xs_val-np.min(Xs_val))/(np.max(Xs_val)-np.min(Xs_val)), (Xs_tst-np.min(Xs_tst))/(np.max(Xs_tst)-np.min(Xs_tst))
Xs_trn, Xs_val, Xs_tst = np.expand_dims(Xs_trn, axis = 3), np.expand_dims(Xs_val, axis = 3), np.expand_dims(Xs_tst, axis = 3)
ys_tst = ys_tst.reshape(-1,1)
ys_trn = ys_trn.reshape(-1,1)
# load target data
if dataset == 'dense':
	nb_target = 7100
elif dataset == 'hetero':
	nb_target = 36000
elif dataset == 'scattered':
	nb_target = 33000
elif dataset == 'fatty':
	nb_target = 9000
elif dataset == 'total':
	nb_target = 85000
Xt_trn, Xt_val, Xt_tst, yt_trn, yt_val, yt_tst = load_target(dataset = dataset, train = nb_target, valid = 100)
Xt_trn, Xt_val, Xt_tst = (Xt_trn-np.min(Xt_trn))/(np.max(Xt_trn)-np.min(Xt_trn)), (Xt_val-np.min(Xt_val))/(np.max(Xt_val)-np.min(Xt_val)), (Xt_tst-np.min(Xt_tst))/(np.max(Xt_tst)-np.min(Xt_tst))
Xt_trn, Xt_val, Xt_tst = np.expand_dims(Xt_trn, axis = 3), np.expand_dims(Xt_val, axis = 3), np.expand_dims(Xt_tst, axis = 3)
yt_trn, yt_val, yt_tst = yt_trn.reshape(-1,1), yt_val.reshape(-1,1), yt_tst.reshape(-1,1)
Xt_trn_l = np.concatenate([Xt_trn[0:nb_trg_labels,:],Xt_trn[nb_target:nb_target+nb_trg_labels,:]], axis = 0)
yt_trn_l = np.concatenate([yt_trn[0:nb_trg_labels,:],yt_trn[nb_target:nb_target+nb_trg_labels,:]], axis = 0)
# DA = '/data/results/{}-{}'.format(os.path.basename(source), os.path.basename(target))
DA = os.path.join(output_folder, '{}-{}'.format(os.path.basename(source), os.path.basename(target)))
generate_folder(DA)
base_model_folder = os.path.join(DA, source_model_name)
generate_folder(base_model_folder)
# copy the source weight file to the DA_model_folder
DA_model_name = 'ADDA2-{0:}-glr-{1:}-dlr-{2:}-bz-{3:}-iter-{4:}-scr-{5:}-shar-{6:}-dis_fc-{7:}-bn-{8:}-tclf-{9:}-sclf-{10:}-tlabels-{11:}-{12:}-cnn-{13:}-dis_bn-{14:}-gcnn-{15:}-smooth-{16:}'.format(dis_param, g_lr,  d_lr, batch_size, nb_steps, source_scratch, shared, dis_fc, den_bn, trg_clf_param, src_clf_param, nb_trg_labels, dataset, dis_cnn, dis_bn, g_cnn, lsmooth)
DA_model_folder = os.path.join(base_model_folder, DA_model_name)
generate_folder(DA_model_folder)
os.system('cp -f {} {}'.format(source_model_file+'*', DA_model_folder))

if source_model_name.split('-')[0] == 'cnn':
	nb_cnn = int(source_model_name.split('-')[1])
else:
	nb_cnn = 4


xs = tf.placeholder("float", shape=[None, 109,109, 1])
ys = tf.placeholder("float", shape=[None, 1])
xt = tf.placeholder("float", shape=[None, 109,109, 1])
yt = tf.placeholder("float", shape=[None, 1])
xt1 = tf.placeholder("float", shape=[None, 109,109, 1])   # input target image with labels
yt1 = tf.placeholder("float", shape=[None, 1])			  # input target image labels

if shared:
	target_scope = 'source'
	target_reuse = True
else:
	target_scope = 'target'
	target_reuse = False

conv_net_src, h_src, source_logit = conv_classifier(xs, nb_cnn = g_cnn, fc_layers = [fc_layer,1],  bn = den_bn, scope_name = 'source')
# flat1 = tf.layers.flatten(conv_net_src)
conv_net_trg, h_trg, target_logit = conv_classifier(xt, nb_cnn = g_cnn, fc_layers = [fc_layer,1],  bn = den_bn, scope_name = target_scope, reuse = target_reuse)
_, _, target_logit_l = conv_classifier(xt1, nb_cnn = g_cnn, fc_layers = [fc_layer,1],  bn = den_bn, scope_name = target_scope, reuse = True)
# flat2 = tf.layers.flatten(conv_net_trg)

source_vars_list = tf.trainable_variables('source')
# source_conv_list = tf.trainable_variables('source/conv')
# source_clf_list = tf.trainable_variables('source/classifier')
source_key_list = [v.name[:-2].replace('source', 'base') for v in tf.trainable_variables('source')]
source_key_direct = {}
for key, var in zip(source_key_list, source_vars_list):
	source_key_direct[key] = var
source_key_direct_except_last_layer = {}
for key, var in zip(source_key_list[:-2], source_vars_list[:-2]):
	source_key_direct_except_last_layer[key] = var
source_saver = tf.train.Saver(source_key_direct, max_to_keep=nb_steps)
pre_trained_saver = tf.train.Saver(source_key_direct_except_last_layer, max_to_keep = nb_steps)

target_vars_list = tf.trainable_variables(target_scope)
target_key_list = [v.name[:-2].replace(target_scope, 'base') for v in tf.trainable_variables(target_scope)]
target_key_direct = {}
for key, var in zip(target_key_list, target_vars_list):
	target_key_direct[key] = var
target_saver = tf.train.Saver(target_key_direct, max_to_keep=nb_steps)
print(target_vars_list)

# source loss
src_clf_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = ys, logits = source_logit))

# mmd loss
# with tf.variable_scope('mmd'):
# 	sigmas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100, 1e3, 1e4, 1e5, 1e6]
# 	gaussian_kernel = partial(gaussian_kernel_matrix, sigmas=tf.constant(sigmas))
# 	loss_value = maximum_mean_discrepancy(h_src, h_trg, kernel=gaussian_kernel)
# 	mmd_loss = dis_param*loss_value
#     mmd_loss = dis_param*tf.maximum(1e-2, loss_value)
if dis_cnn > 0:
	src_logits = discriminator(conv_net_src, nb_cnn = dis_cnn, fc_layers = [128, 1], bn = dis_bn)
	trg_logits = discriminator(conv_net_trg, nb_cnn = dis_cnn, fc_layers = [128, 1], bn = dis_bn, reuse = True)
else:
	src_logits = discriminator(h_src, nb_cnn = 0, fc_layers = [dis_fc, 1], bn = dis_bn)
	trg_logits = discriminator(h_trg, nb_cnn = 0, fc_layers = [dis_fc, 1], bn = dis_bn, reuse = True)

if lsmooth:
	disc_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=src_logits,labels=tf.ones_like(src_logits)-0.1) + tf.nn.sigmoid_cross_entropy_with_logits(logits=trg_logits, labels=tf.zeros_like(trg_logits)+0.1))
else:
	disc_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=src_logits,labels=tf.ones_like(src_logits)) + tf.nn.sigmoid_cross_entropy_with_logits(logits=trg_logits, labels=tf.zeros_like(trg_logits)))
gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=src_logits,labels=tf.zeros_like(src_logits)) + tf.nn.sigmoid_cross_entropy_with_logits(logits=trg_logits, labels=tf.ones_like(trg_logits)))
total_loss_no_labels = dis_param* gen_loss + src_clf_param*src_clf_loss
total_loss = total_loss_no_labels
discr_vars_list = tf.trainable_variables('discriminator')

# total_loss = mmd_loss + src_clf_param*src_clf_loss

if nb_trg_labels > 0:
	trg_clf_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = yt1, logits = target_logit_l))
	total_loss = total_loss + trg_clf_param*trg_clf_loss

disc_step = tf.train.AdamOptimizer(d_lr).minimize(disc_loss, var_list=discr_vars_list)
gen_step = tf.train.AdamOptimizer(g_lr).minimize(total_loss, var_list=target_vars_list)
gen_step_no_labels = tf.train.AdamOptimizer(g_lr).minimize(total_loss_no_labels, var_list=target_vars_list)

# if not shared:
# 	# weight loss
# 	a_list = []
# 	b_list = []
# 	with tf.variable_scope('weight_regulizer'):
# 		for i in range(nb_cnn+2):
# 			a_list.append(tf.Variable(1.0, name='a_{}'.format(i)))
# 			b_list.append(tf.Variable(0.0, name='b_{}'.format(i)))
# 	# source kernel and target kernel
# 	source_kernels = [v for v in tf.trainable_variables('source') if 'kernel' in v.name]
# 	target_kernels = [v for v in tf.trainable_variables('target') if 'kernel' in v.name]
# 	source_bias = [v for v in tf.trainable_variables('source') if 'bias' in v.name]
# 	target_bias = [v for v in tf.trainable_variables('target') if 'bias' in v.name]
# 	layer_loss_list = []
# 	for a, b, sk, tk, sb, tb in list(zip(a_list[:-1], b_list[:-1], source_kernels[:-1], target_kernels[:-1], source_bias[:-1], target_bias[:-1])):
# 		layer_loss_list.append(tm.exp(tf.nn.l2_loss(tm.scalar_mul(a, sk) + b - tk)) -1)
# 		layer_loss_list.append(tm.exp(tf.nn.l2_loss(tm.scalar_mul(a, sb) + b - tb)) -1)
# 	# 	layer_loss_list.append(tm.exp(tf.nn.l2_loss(tm.subtract(tm.add(tm.scalar_mul(a, sb), b), tb)))-1)
# 	# source bais and target bais
# 	w_loss = tf.add_n(layer_loss_list)
# 	total_loss = total_loss + w_loss
# 	gen_step = tf.train.AdamOptimizer(lr).minimize(total_loss, var_list = target_vars_list + source_vars_list + tf.trainable_variables('weight_regulizer'))
# else:
# gen_step = tf.train.AdamOptimizer(lr).minimize(total_loss, var_list = target_vars_list)

D_loss_list = []
G_loss_list = []
sC_loss_list = []
tC_loss_list = []
test_auc_list = []
val_auc_list = []
dom_acc_list = []
train_auc_list = []
best_val_auc = 0
train_target_AUC = 0.5
tC_loss = 1.4

## model loading verification
# with tf.Session() as sess:
# 	tf.global_variables_initializer().run(session=sess)
# # 	pre_trained_saver.restore(sess, source_model_file)
# 	source_saver.restore(sess, source_model_file)
# 	if not shared:
# 		target_saver.restore(sess, source_model_file)
# 	# source to source (target loading)
# 	print_yellow('>>>>>> Check the Initial Source Model Loading <<<<<<')
# 	test_source_logit_source = source_logit.eval(session=sess,feed_dict={xs:Xs_tst})
# 	test_source_stat_source = np.exp(test_source_logit_source)
# 	test_source_AUC_source = roc_auc_score(ys_tst, test_source_stat_source)
# 	print_yellow('Source loading: source-source:{0:.4f} '.format(test_source_AUC_source))
# 	# source to source (target loading)
# 	test_source_logit = target_logit.eval(session=sess,feed_dict={xt:Xs_tst})
# 	test_source_stat = np.exp(test_source_logit)
# 	test_source_AUC = roc_auc_score(ys_tst, test_source_stat)
# 	# source to target (target loading)
# 	test_target_logit = target_logit.eval(session=sess,feed_dict={xt:Xt_tst})
# 	test_target_stat = np.exp(test_target_logit)
# 	test_target_AUC = roc_auc_score(yt_tst, test_target_stat)
# 	print_yellow('Target loading: source-source:{0:.4f} source-target {1:.4f}'.format(test_source_AUC, test_target_AUC))

nd_step_used = nd_steps
ng_step_used = ng_steps
# sess = tf.Session()
with tf.Session() as sess:
	tf.global_variables_initializer().run(session=sess)
	if not source_scratch:
# 		pre_trained_saver.restore(sess, source_model_file)
		target_saver.restore(sess, source_model_file)
		if not shared:
			target_saver.restore(sess, source_model_file)
	for iteration in range(nb_steps):
		indices_s = np.random.randint(0, Xs_trn.shape[0]-1, batch_size)
		batch_s = Xs_trn[indices_s,:]
		batch_ys = ys_trn[indices_s,:]
		indices_t = np.random.randint(0, Xt_trn.shape[0]-1, batch_size)
		batch_t = Xt_trn[indices_t,:]
# 		for _ in range(nd_steps):
		_, D_loss = sess.run([disc_step, disc_loss], feed_dict={xs: batch_s, xt: batch_t})
# 		for _ in range(ng_steps):
		test_source_logit = source_logit.eval(session=sess,feed_dict={xs:Xs_tst})
		test_source_stat = np.exp(test_source_logit)
		test_source_AUC = roc_auc_score(ys_tst, test_source_stat)
# 		if nb_trg_labels > 0 and test_source_AUC>0.8:
		if nb_trg_labels > 0:
			indices_tl = np.random.randint(0, 2*nb_trg_labels-1, 100)
			batch_xt_l, batch_yt_l = Xt_trn_l[indices_tl, :], yt_trn_l[indices_tl, :]
			_, G_loss, sC_loss, tC_loss, trg_digit = sess.run([gen_step, gen_loss, src_clf_loss, trg_clf_loss, target_logit_l], feed_dict={xs: batch_s, xt: batch_t, ys: batch_ys, xt1:batch_xt_l, yt1:batch_yt_l})
			train_target_stat = np.exp(trg_digit)
			train_target_AUC = roc_auc_score(batch_yt_l, train_target_stat)
		else:
			_, G_loss, sC_loss = sess.run([gen_step_no_labels, gen_loss, src_clf_loss], feed_dict={xs: batch_s, xt: batch_t, ys: batch_ys})
		#training
		train_source_logit = src_logits.eval(session=sess,feed_dict={xs:batch_s})
		train_target_logit = trg_logits.eval(session=sess,feed_dict={xt:batch_t})
		domain_preds = np.concatenate([train_source_logit, train_target_logit], axis = 0) > 0
		domain_labels = np.concatenate([np.ones(train_source_logit.shape), np.zeros(train_target_logit.shape)])
		domain_acc = np.sum(domain_preds == domain_labels)/domain_preds.shape[0]
		dom_acc_list.append(domain_acc)
# 		if domain_acc > acc_up:
# # 			nd_step_used = 1
# 			ng_step_used = 10
# 		elif domain_acc < acc_down:
# 			nd_step_used = 10
# 		else:
# 			ng_step_used = 1
# 			nd_step_used = 1
# 		elif domain_acc > dAcc2:
# 			nd_step_used = 0
# 			ng_step_used = 10
# 		else:
# 			nd_step_used = nd_steps
# 			ng_step_used = ng_steps
		#testing
		test_target_logit = target_logit.eval(session=sess,feed_dict={xt:Xt_tst})
		test_target_stat = np.exp(test_target_logit)
		test_target_AUC = roc_auc_score(yt_tst, test_target_stat)
		val_target_logit = target_logit.eval(session=sess,feed_dict={xt:Xt_val})
		val_target_stat = np.exp(val_target_logit)
		val_target_AUC = roc_auc_score(yt_val, val_target_stat)
		test_auc_list.append(test_target_AUC)
		val_auc_list.append(val_target_AUC)
		G_loss_list.append(G_loss)
		D_loss_list.append(D_loss)
		sC_loss_list.append(sC_loss)
		# save results
		np.savetxt(os.path.join(DA_model_folder,'test_auc.txt'), test_auc_list)
		np.savetxt(os.path.join(DA_model_folder,'val_auc.txt'), val_auc_list)
		np.savetxt(os.path.join(DA_model_folder,'dom_acc.txt'), dom_acc_list)
		np.savetxt(os.path.join(DA_model_folder,'D_loss.txt'),D_loss_list)
		np.savetxt(os.path.join(DA_model_folder,'G_loss.txt'),G_loss_list)
		np.savetxt(os.path.join(DA_model_folder,'src_clf_loss.txt'),sC_loss_list)
		# print and plot results
		print_block(symbol = '-', nb_sybl = 60)
		print_yellow(os.path.basename(DA_model_folder))
		if nb_trg_labels > 0:
			train_auc_list.append(train_target_AUC)
			tC_loss_list.append(tC_loss)
			np.savetxt(os.path.join(DA_model_folder,'train_auc.txt'), train_auc_list)
			np.savetxt(os.path.join(DA_model_folder,'trg_clf_loss.txt'),tC_loss_list)
			print_green('AUC: T-test {0:.4f}, T-valid {1:.4f}, T-train {2:.4f}, S-test: {3:.4f}; ACC: dom {4:.4f}'.format(test_target_AUC, val_target_AUC, train_target_AUC, test_source_AUC, domain_acc))
			print_yellow('Loss: D:{0:.4f}, G:{1:.4f}, S:{2:.4f}, T:{3:.4f}, Iter:{4:}'.format(D_loss, G_loss, sC_loss, tC_loss, iteration))
			plot_LOSS(DA_model_folder+'/loss_{}.png'.format(DA_model_name), G_loss_list, sC_loss_list, tC_loss_list)
			plot_loss(DA_model_folder, D_loss_list, G_loss_list, DA_model_folder+'/adver_{}.png'.format(DA_model_name))
			plot_AUCs_DomACC(DA_model_folder+'/AUC_dom_{}.png'.format(DA_model_name), train_auc_list, val_auc_list, test_auc_list, dom_acc_list)
			plot_AUCs(DA_model_folder+'/AUC_{}.png'.format(DA_model_name), train_auc_list, val_auc_list, test_auc_list)
		else:
			print_green('AUC: T-test {0:.4f}, T-valid {1:.4f}, S-test: {2:.4f}; ACC: dom {3:.4f}'.format(test_target_AUC, val_target_AUC, test_source_AUC, domain_acc))
			print_yellow('Loss: D:{0:.4f}, G:{1:.4f}, S:{2:.4f}, Iter:{3:}'.format(D_loss, G_loss, sC_loss, iteration))
			plot_loss(DA_model_folder, G_loss_list, sC_loss_list, DA_model_folder+'/loss_{}.png'.format(DA_model_name))
			plot_loss(DA_model_folder, D_loss_list, G_loss_list, DA_model_folder+'/adver_{}.png'.format(DA_model_name))
			plot_auc_dom_acc_iterations(test_auc_list, val_auc_list, dom_acc_list, DA_model_folder+'/AUC_dom_{}.png'.format(DA_model_name))
			plot_auc_iterations(test_auc_list, val_auc_list, DA_model_folder+'/AUC_{}.png'.format(DA_model_name))
		# save models
		if iteration%100==0:
			target_saver.save(sess, DA_model_folder +'/target', global_step= iteration)
		if best_val_auc < val_target_AUC:
			best_val_auc = val_target_AUC
			target_saver.save(sess, DA_model_folder+'/target_best')
			np.savetxt(os.path.join(DA_model_folder,'test_stat.txt'), test_target_stat)
			np.savetxt(os.path.join(DA_model_folder,'test_best_auc.txt'), [test_target_AUC])
			print_red('Update best:'+DA_model_folder)
