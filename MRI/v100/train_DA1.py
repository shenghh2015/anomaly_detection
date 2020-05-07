import tensorflow as tf

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

# generate the folder
def generate_folder(folder):
    import os
    if not os.path.exists(folder):
        os.makedirs(folder)

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
parser.add_argument("--dis_cnn", type=int)
parser.add_argument("--dis_fc", type=int)
parser.add_argument("--dis_bn", type=bool)
parser.add_argument("--D_lr", type = float)
parser.add_argument("--G_lr", type = float)
parser.add_argument("--nD", type = int)
parser.add_argument("--nG", type = int)
parser.add_argument("--dAcc1", type = float)
parser.add_argument("--dAcc2", type = float)
parser.add_argument("--iters", type = int)
parser.add_argument("--bz", type = int)
parser.add_argument("--lamda", type = float)


args = parser.parse_args()
gpu_num = args.gpu
dis_cnn = args.dis_cnn
dis_fc = args.dis_fc
dis_bn = args.dis_bn
batch_size = args.bz
d_lr = args.D_lr
g_lr = args.G_lr
nb_steps = args.iters
nd_steps = args.nD
ng_steps = args.nG
dAcc1 = args.dAcc1
dAcc2 = args.dAcc2
lmd = args.lamda

if False:
	gpu_num = 6
	batch_size = 400
	nb_dis = 5
	d_lr = 1e-5
	g_lr = 1e-6
	nb_steps = 1000
	nd_steps = 10
	ng_steps = 10
	dis_cnn = 2
	dis_bn = True
	dis_fc = 128
	dAcc1 = args.dAcc1
	dAcc2 = args.dAcc2
	lmd = 1.0

os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)
# hyper-parameters
noise = 2.0
sig_rate = 0.035
source_model_name = 'cnn-4-bn-True-noise-2.0-trn-100000-sig-0.035-bz-400-lr-5e-05-Adam-4.0k'
# load source data
source = '/data/results/CLB'
target = '/data/results/FDA'
source_model_file = os.path.join(source, source_model_name, 'source-best')

# load source data
nb_source = 100000
Xs_trn, Xs_val, Xs_tst, ys_trn, ys_val, ys_tst = load_source(train = nb_source, sig_rate = sig_rate)
Xs_trn, Xs_val, Xs_tst = np.random.RandomState(2).normal(Xs_trn, noise), np.random.RandomState(0).normal(Xs_val, noise), np.random.RandomState(1).normal(Xs_tst, noise)
Xs_trn, Xs_val, Xs_tst = (Xs_trn-np.min(Xs_trn))/(np.max(Xs_trn)-np.min(Xs_trn)), (Xs_val-np.min(Xs_val))/(np.max(Xs_val)-np.min(Xs_val)), (Xs_tst-np.min(Xs_tst))/(np.max(Xs_tst)-np.min(Xs_tst))
Xs_trn, Xs_val, Xs_tst = np.expand_dims(Xs_trn, axis = 3), np.expand_dims(Xs_val, axis = 3), np.expand_dims(Xs_tst, axis = 3)
ys_trn = ys_trn.reshape(-1,1)
ys_tst = ys_tst.reshape(-1,1)
# load target data
nb_target = 85000
Xt_trn, Xt_val, Xt_tst, _, yt_val, yt_tst = load_target(dataset = 'total', train = nb_target)
Xt_trn, Xt_val, Xt_tst = (Xt_trn-np.min(Xt_trn))/(np.max(Xt_trn)-np.min(Xt_trn)), (Xt_val-np.min(Xt_val))/(np.max(Xt_val)-np.min(Xt_val)), (Xt_tst-np.min(Xt_tst))/(np.max(Xt_tst)-np.min(Xt_tst))
Xt_trn, Xt_val, Xt_tst = np.expand_dims(Xt_trn, axis = 3), np.expand_dims(Xt_val, axis = 3), np.expand_dims(Xt_tst, axis = 3)
yt_tst = yt_tst.reshape(-1,1)

DA = '/data/results/{}-{}'.format(os.path.basename(source), os.path.basename(target))
generate_folder(DA)
base_model_folder = os.path.join(DA, source_model_name)
generate_folder(base_model_folder)
# copy the source weight file to the DA_model_folder
DA_model_name = 'shared-cnn-{0:}-fc-{1:}-bn-{2:}-bz-{3:}-D_lr-{4:}-G_lr-{5:}-nD-{6:}-nG-{7:}-iter-{8:}-ac1-{9:}-ac2-{10:}-lmd-{11:}'.format(dis_cnn, dis_fc, dis_bn, batch_size, d_lr, g_lr, nd_steps, ng_steps, nb_steps, dAcc1, dAcc2, lmd)
DA_model_folder = os.path.join(base_model_folder, DA_model_name)
generate_folder(DA_model_folder)
os.system('cp -f {} {}'.format(source_model_file+'*', DA_model_folder))

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
xt = tf.placeholder("float", shape=[None, 109,109, 1])
yt = tf.placeholder("float", shape=[None, 1])

conv_net_src, h_src, source_logit = conv_classifier(xs, nb_cnn = nb_cnn, fc_layers = [128,1],  bn = bn, scope_name = 'source')
conv_net_trg, h_trg, target_logit = conv_classifier(xt, nb_cnn = nb_cnn, fc_layers = [128,1],  bn = bn, scope_name = 'source', reuse = True)

source_vars_list = tf.trainable_variables('source')
source_key_list = [v.name[:-2].replace('source', 'base') for v in tf.trainable_variables('source')]
source_key_direct = {}
for key, var in zip(source_key_list, source_vars_list):
	source_key_direct[key] = var
source_saver = tf.train.Saver(source_key_direct, max_to_keep=nb_steps)

# target_vars_list = tf.trainable_variables('target')
# target_key_list = [v.name[:-2].replace('target', 'base') for v in tf.trainable_variables('target')]
# target_key_direct = {}
# for key, var in zip(target_key_list, target_vars_list):
# 	target_key_direct[key] = var
# target_saver = tf.train.Saver(target_key_direct, max_to_keep=nb_steps)

if dis_cnn > 0:
	src_logits = discriminator(conv_net_src, nb_cnn = dis_cnn, fc_layers = [128, 1], bn = dis_bn)
	trg_logits = discriminator(conv_net_trg, nb_cnn = dis_cnn, fc_layers = [128, 1], bn = dis_bn, reuse = True)
else:
	src_logits = discriminator(h_src, nb_cnn = 0, fc_layers = [dis_fc, 1], bn = dis_bn)
	trg_logits = discriminator(h_trg, nb_cnn = 0, fc_layers = [dis_fc, 1], bn = dis_bn, reuse = True)

clf_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=source_logit, labels= ys))
disc_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=src_logits, labels=tf.ones_like(src_logits)) + tf.nn.sigmoid_cross_entropy_with_logits(logits= trg_logits, labels=tf.zeros_like(trg_logits)))
gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=src_logits, labels=tf.zeros_like(src_logits)) + tf.nn.sigmoid_cross_entropy_with_logits(logits=trg_logits,labels=tf.ones_like(trg_logits)))
total_loss = clf_loss + lmd*gen_loss
discr_vars_list = tf.trainable_variables('discriminator')
disc_step = tf.train.AdamOptimizer(d_lr).minimize(disc_loss, var_list= discr_vars_list)
gen_step = tf.train.AdamOptimizer(g_lr).minimize(total_loss, var_list = source_vars_list)

D_loss_list = []
G_loss_list = []
test_auc_list = []
val_auc_list = []
# d_lr = 1e-5
# g_lr = 1e-5
# nb_steps = 1000
# nd_steps = 10
# ng_steps = 10

## model loading verification
with tf.Session() as sess:
	tf.global_variables_initializer().run(session=sess)
	source_saver.restore(sess, source_model_file)
# 	target_saver.restore(sess, source_model_file)
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

nd_step_used = nd_steps
ng_step_used = ng_steps
with tf.Session() as sess:
	tf.global_variables_initializer().run(session=sess)
	source_saver.restore(sess, source_model_file)
# 	target_saver.restore(sess, source_model_file)
	for iteration in range(nb_steps):
		indices_s = np.random.randint(0, Xs_trn.shape[0], batch_size)
		batch_s = Xs_trn[indices_s,:]
		batch_ys = ys_trn[indices_s,:]
		indices_t = np.random.randint(0, Xt_trn.shape[0], batch_size)
		batch_t = Xt_trn[indices_t,:]
		for _ in range(nd_step_used):
			_, D_loss = sess.run([disc_step, disc_loss], feed_dict={xs: batch_s, xt: batch_t})
		for _ in range(ng_step_used):
			_, G_loss, C_loss = sess.run([gen_step, gen_loss, clf_loss], feed_dict={xs: batch_s, xt: batch_t, ys: batch_ys})
		#training
		train_source_logit = src_logits.eval(session=sess,feed_dict={xs:batch_s})
		train_target_logit = trg_logits.eval(session=sess,feed_dict={xt:batch_t})
		domain_preds = np.concatenate([train_source_logit, train_target_logit], axis = 0) > 0
		domain_labels = np.concatenate([np.ones(train_source_logit.shape), np.zeros(train_target_logit.shape)])
		domain_acc = np.sum(domain_preds == domain_labels)/domain_preds.shape[0]
		if domain_acc > dAcc1:
			nd_step_used = 1
			ng_step_used = 5
		elif domain_acc > dAcc2:
			nd_step_used = 0
			ng_step_used = 5
		else:
			nd_step_used = nd_steps
			ng_step_used = ng_steps		
		#testing
		test_source_logit = source_logit.eval(session=sess,feed_dict={xs:Xs_tst})
		test_source_stat = np.exp(test_source_logit)
		test_source_AUC = roc_auc_score(ys_tst, test_source_stat)
		test_target_logit = target_logit.eval(session=sess,feed_dict={xt:Xt_tst})
		test_target_stat = np.exp(test_target_logit)
		test_target_AUC = roc_auc_score(yt_tst, test_target_stat)
		val_target_logit = target_logit.eval(session=sess,feed_dict={xt:Xt_val})
		val_target_stat = np.exp(val_target_logit)
		val_target_AUC = roc_auc_score(yt_val, val_target_stat)
		# print results
		print_block(symbol = '-', nb_sybl = 60)
		print_green('AUC: T-test {0:.4f}, T-valid {1:.4f}; S-test: {2:.4f}'.format(test_target_AUC, val_target_AUC, test_source_AUC))
		print_yellow('Loss: D {0:.4f}, G :{1:.4f} C: {2:.4f}; Dom acc: {3:.4f} Iter:{4}'.format(D_loss, G_loss, C_loss, domain_acc, iteration))
		# save results
		D_loss_list.append(D_loss)
		G_loss_list.append(G_loss)
		test_auc_list.append(test_target_AUC)
		val_auc_list.append(val_target_AUC)
		print_yellow(os.path.basename(DA_model_folder))
		plot_loss(DA_model_folder, D_loss_list, G_loss_list, DA_model_folder+'/loss_{}.png'.format(DA_model_name))
		np.savetxt(os.path.join(DA_model_folder,'test_auc.txt'), test_auc_list)
		np.savetxt(os.path.join(DA_model_folder,'val_auc.txt'), val_auc_list)
		np.savetxt(os.path.join(DA_model_folder,'D_loss.txt'),D_loss_list)
		np.savetxt(os.path.join(DA_model_folder,'G_loss.txt'),G_loss_list)
		plot_auc_iterations(test_auc_list, val_auc_list, DA_model_folder+'/AUC_{}.png'.format(DA_model_name))
		# save models
		source_saver.save(sess, DA_model_folder +'/target')
