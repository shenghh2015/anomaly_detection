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
def plot_loss(model_name,loss,val_loss, file_name):
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
parser.add_argument("--gpu", type=int)
parser.add_argument("--docker", type = str2bool, default = True)
# parser.add_argument("--shared", type = str2bool, default = True)
parser.add_argument("--lr", type = float)
parser.add_argument("--iters", type = int)
parser.add_argument("--bz", type = int)
# parser.add_argument("--mmd_param", type = float, default = 1.0)
# parser.add_argument("--trg_clf_param", type = float, default = 1.0)
# parser.add_argument("--src_clf_param", type = float, default = 1.0)
parser.add_argument("--source_scratch", type = str2bool, default = False)
parser.add_argument("--nb_trg_labels", type = int, default = 0)
parser.add_argument("--fc_layer", type = int, default = 128)
parser.add_argument("--DA_FLAG", type = str2bool, default = False)
parser.add_argument("--source_name", type = str, default = 'cnn-4-bn-False-noise-2.0-trn-100000-sig-0.035-bz-400-lr-5e-05-Adam-100.0k')
parser.add_argument("--DA_name", type = str, default = 'mmd-1.0-lr-0.0001-bz-400-iter-50000-scr-True-shar-True-fc-128-bn-False-tclf-0.5-sclf-1.0-trg_labels-0')
parser.add_argument("--clf_v", type = int, default = 1)
parser.add_argument("--dataset", type = str, default = 'total')
parser.add_argument("--valid", type = int, default = 400)
# parser.add_argument("--den_bn", type = str2bool, default = False)

args = parser.parse_args()
print(args)

gpu_num = args.gpu
docker = args.docker
# shared = args.shared
# shared = True
batch_size = args.bz
nb_steps = args.iters
# mmd_param = args.mmd_param
lr = args.lr
nb_trg_labels = args.nb_trg_labels
source_scratch = args.source_scratch
# source_scratch = False
fc_layer = args.fc_layer
den_bn = False
DA_FLAG = args.DA_FLAG
source_model_name = args.source_name
DA_name = args.DA_name
clf_v = args.clf_v
dataset = args.dataset
valid = args.valid
# trg_clf_param = args.trg_clf_param
# src_clf_param = args.src_clf_param

if False:
	gpu_num = 1
	lr = 1e-5
	batch_size = 400
	nb_steps = 1000
	mmd_param = 1.0
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
# load source data
# source = '/data/results/CLB'
# target = '/data/results/FDA'

print(output_folder)
# hyper-parameters
noise = 2.0
sig_rate = 0.035
# source_model_name = 'cnn-4-bn-True-noise-2.0-trn-100000-sig-0.035-bz-400-lr-5e-05-Adam-4.0k'
if not DA_FLAG:
	source = os.path.join(output_folder,'CLB')
	target = os.path.join(output_folder,'FDA')
	source_model_file = os.path.join(source, source_model_name, 'source-best')
	DA = os.path.join(output_folder, '{}-{}'.format(os.path.basename(source), os.path.basename(target)))
	generate_folder(DA)
	base_model_folder = os.path.join(DA, source_model_name)
	generate_folder(base_model_folder)
	DA_model_name = 'TF-lr-{0:}-bz-{1:}-iter-{2:}-scr-{3:}-fc-{4:}-bn-{5:}-trg_labels-{6:}-clf_v{7:}-{8:}-valid-{9:}'.format(lr, batch_size, nb_steps, source_scratch, fc_layer, den_bn, nb_trg_labels, clf_v, dataset, valid)
	DA_model_folder = os.path.join(base_model_folder, DA_model_name)
	generate_folder(DA_model_folder)
	os.system('cp -f {} {}'.format(source_model_file+'*', DA_model_folder))
# 	source_model_name = 'cnn-4-bn-False-noise-2.0-trn-100000-sig-0.035-bz-400-lr-5e-05-Adam-100.0k'
else:
	source = os.path.join(output_folder,'CLB-FDA')
	target = os.path.join(output_folder,'DA-TF')
	source_model_file = os.path.join(source, source_model_name, DA_name, 'target_best')
	DA = target
	generate_folder(DA)
	base_model_folder = os.path.join(DA, DA_name)
	generate_folder(base_model_folder)
	DA_model_name = 'DA-TF-lr-{0:}-bz-{1:}-iter-{2:}-scr-{3:}-fc-{4:}-bn-{5:}-trg_labels-{6:}'.format(lr, batch_size, nb_steps, source_scratch, fc_layer, den_bn, nb_trg_labels)
	DA_model_folder = os.path.join(base_model_folder, DA_model_name)
	generate_folder(DA_model_folder)
	os.system('cp -f {} {}'.format(source_model_file+'*', DA_model_folder))

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
	sp_offset = 7100
elif dataset == 'hetero':
	nb_target = 36000
	sp_offset = 36000
elif dataset == 'scattered':
	nb_target = 33000
	sp_offset = 33000	
elif dataset == 'fatty':
	nb_target = 9000
	sp_offset = 9000	
elif dataset == 'total':
	nb_target = 85000
	sp_offset = 85000
Xt_trn, Xt_val, Xt_tst, yt_trn, yt_val, yt_tst = load_target(dataset = dataset, train = nb_target)
Xt_trn, Xt_val, Xt_tst = (Xt_trn-np.min(Xt_trn))/(np.max(Xt_trn)-np.min(Xt_trn)), (Xt_val-np.min(Xt_val))/(np.max(Xt_val)-np.min(Xt_val)), (Xt_tst-np.min(Xt_tst))/(np.max(Xt_tst)-np.min(Xt_tst))
Xt_trn, Xt_val, Xt_tst = np.expand_dims(Xt_trn, axis = 3), np.expand_dims(Xt_val, axis = 3), np.expand_dims(Xt_tst, axis = 3)
yt_trn, yt_val, yt_tst = yt_trn.reshape(-1,1), yt_val.reshape(-1,1), yt_tst.reshape(-1,1)
Xt_trn_l = np.concatenate([Xt_trn[0:nb_trg_labels,:],Xt_trn[sp_offset:sp_offset+nb_trg_labels,:]], axis = 0)
yt_trn_l = np.concatenate([yt_trn[0:nb_trg_labels,:],yt_trn[sp_offset:sp_offset+nb_trg_labels,:]], axis = 0)

if source_model_name.split('-')[0] == 'cnn':
	nb_cnn = int(source_model_name.split('-')[1])
else:
	nb_cnn = 4

if source_model_name.split('-')[2] == 'bn':
	bn = bool(source_model_name.split('-')[3])
else:
	bn = False

xt = tf.placeholder("float", shape=[None, 109,109, 1])
yt = tf.placeholder("float", shape=[None, 1])

target_scope = 'target'
if clf_v == 2:
	conv_net_trg, h_trg, target_logit = conv_classifier2(xt, nb_cnn = nb_cnn, fc_layers = [fc_layer,1],  bn = den_bn, scope_name = target_scope)
else:
	conv_net_trg, h_trg, target_logit = conv_classifier(xt, nb_cnn = nb_cnn, fc_layers = [fc_layer,1],  bn = den_bn, scope_name = target_scope)
target_vars_list = tf.trainable_variables(target_scope)
target_key_list = [v.name[:-2].replace(target_scope, 'base') for v in tf.trainable_variables(target_scope)]
target_key_direct = {}
for key, var in zip(target_key_list, target_vars_list):
	target_key_direct[key] = var
target_saver = tf.train.Saver(target_key_direct, max_to_keep=nb_steps)

trg_clf_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = yt, logits = target_logit))
gen_step = tf.train.AdamOptimizer(lr).minimize(trg_clf_loss, var_list = target_vars_list)

C_loss_list = []
test_auc_list = []
val_auc_list = []
train_auc_list = []
best_val_auc = 0

## model loading verification
with tf.Session() as sess:
	tf.global_variables_initializer().run(session=sess)
# 	pre_trained_saver.restore(sess, source_model_file)
	target_saver.restore(sess, source_model_file)
	# source to source (target loading)
	print_yellow('>>>>>> Check the Initial Source Model Loading <<<<<<')
	# source to source (target loading)
	test_source_logit = target_logit.eval(session=sess,feed_dict={xt:Xs_tst})
	test_source_stat = np.exp(test_source_logit)
	test_source_AUC = roc_auc_score(ys_tst, test_source_stat)
	# source to target (target loading)
	test_target_logit = target_logit.eval(session=sess,feed_dict={xt:Xt_tst})
	test_target_stat = np.exp(test_target_logit)
	test_target_AUC = roc_auc_score(yt_tst, test_target_stat)
	print_yellow('Target loading: source-source:{0:.4f} source-target {1:.4f}'.format(test_source_AUC, test_target_AUC))

# nd_step_used = nd_steps
# ng_step_used = ng_steps
# sess = tf.Session()
with tf.Session() as sess:
	tf.global_variables_initializer().run(session=sess)
	if not source_scratch:
		target_saver.restore(sess, source_model_file)
	for iteration in range(nb_steps):
		indices_tl = np.random.randint(0, 2*nb_trg_labels-1, batch_size)
		batch_xt_l, batch_yt_l = Xt_trn_l[indices_tl, :], yt_trn_l[indices_tl, :]
		_, C_loss, trg_digit = sess.run([gen_step, trg_clf_loss, target_logit], feed_dict={xt: batch_xt_l, yt: batch_yt_l})
		train_target_stat = np.exp(trg_digit)
		train_target_AUC = roc_auc_score(batch_yt_l, train_target_stat)
		test_target_logit = target_logit.eval(session=sess,feed_dict={xt:Xt_tst})
		test_target_stat = np.exp(test_target_logit)
		test_target_AUC = roc_auc_score(yt_tst, test_target_stat)
		val_target_logit = target_logit.eval(session=sess,feed_dict={xt:Xt_val})
		val_target_stat = np.exp(val_target_logit)
		val_target_AUC = roc_auc_score(yt_val, val_target_stat)
		# print results
		print_block(symbol = '-', nb_sybl = 60)
		print_green('AUC: T-test {0:.4f}, T-valid {1:.4f}, T-train-{2:.4f}; S-test: {3:.4f}'.format(test_target_AUC, val_target_AUC, train_target_AUC, test_source_AUC))
		print_yellow('C loss :{0:.4f}, Iter:{1:}'.format(C_loss, iteration))
		# save results
		C_loss_list.append(C_loss)
		test_auc_list.append(test_target_AUC)
		val_auc_list.append(val_target_AUC)
		train_auc_list.append(train_target_AUC)
		print_yellow(os.path.basename(DA_model_folder))
		plot_loss(DA_model_folder, C_loss_list, C_loss_list, DA_model_folder+'/loss_{}.png'.format(DA_model_name))
		np.savetxt(os.path.join(DA_model_folder,'clf_loss.txt'),C_loss_list)
		np.savetxt(os.path.join(DA_model_folder,'train_auc.txt'), test_auc_list)
		np.savetxt(os.path.join(DA_model_folder,'test_auc.txt'), test_auc_list)
		np.savetxt(os.path.join(DA_model_folder,'val_auc.txt'), val_auc_list)
		np.savetxt(os.path.join(DA_model_folder,'clf_loss.txt'),C_loss_list)
# 		plot_auc_iterations(test_auc_list, val_auc_list, DA_model_folder+'/AUC_{}.png'.format(DA_model_name))
		plot_AUCs(DA_model_folder+'/AUC_{}.png'.format(DA_model_name), train_auc_list, val_auc_list, test_auc_list)
		# save models
		if iteration%100 == 0:
			target_saver.save(sess, DA_model_folder +'/target', global_step= iteration)
		if best_val_auc < val_target_AUC:
			best_val_auc = val_target_AUC
			target_saver.save(sess, DA_model_folder+'/target_best')
			np.savetxt(os.path.join(DA_model_folder,'test_stat.txt'), test_target_stat)
			np.savetxt(os.path.join(DA_model_folder,'test_best_auc.txt'), [test_target_AUC])
			print_red('Update best:'+DA_model_folder)