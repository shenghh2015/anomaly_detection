import tensorflow as tf

import numpy as np
import os
import argparse

from load_data import *
from model import *

def str2bool(value):
    return value.lower() == 'true'

# generate the folder
def generate_folder(folder):
    import os
    if not os.path.exists(folder):
        os.system('mkdir {}'.format(folder))

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

## input parameters
parser = argparse.ArgumentParser()
parser.add_argument("--gpu_num", type=int)
parser.add_argument("--nb_cnn", type = int)
parser.add_argument("--bn", type = str2bool, default = False)
parser.add_argument("--lr", type = float)
parser.add_argument("--nb_train", type = int)
parser.add_argument("--noise", type = float)
parser.add_argument("--sig_rate", type = float)
parser.add_argument("--bz", type = int)
parser.add_argument("--optimizer", type = str)
parser.add_argument("--nb_steps", type = int, default = 100000)

args = parser.parse_args()
gpu_num = args.gpu_num
nb_cnn = args.nb_cnn
bn = args.bn
lr = args.lr
nb_train = args.nb_train
noise = args.noise
sig_rate = args.sig_rate
batch_size = args.bz
optimizer = args.optimizer
num_steps = args.nb_steps
# gpu_num = 1
# nb_train = 100000
# sig_rate = 0.035
# noise = 2
# nb_cnn = 4
# bn = False
# batch_size = 200
# lr = 5e-5
# num_steps = 1000
# optimizer="Adam"
# l2_param = 1e-5
fc_layers = [128,1]


os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)

## load data
X_trn, X_val, X_tst, y_trn, y_val, y_tst = load_source(train = nb_train, sig_rate = sig_rate) 							# train, valid, test data
X_val, X_tst = np.random.RandomState(0).normal(X_val, noise), np.random.RandomState(1).normal(X_tst, noise) 			# add noise
X_val, X_tst = (X_val-np.min(X_val))/(np.max(X_val)-np.min(X_val)), (X_tst-np.min(X_tst))/(np.max(X_tst)-np.min(X_tst)) # data normalization
X_val, X_tst = np.expand_dims(X_val, axis = 3), np.expand_dims(X_tst, axis = 3)
y_val, y_tst = y_val.reshape(-1,1), y_tst.reshape(-1,1)

#model_root_folder = 'data/CLB' 	# dataset
model_root_folder = '/data/results/CLB'
generate_folder(model_root_folder)

direct = os.path.join(model_root_folder,'cnn-{}-bn-{}-noise-{}-trn-{}-sig-{}-bz-{}-lr-{}-{}-stp-{}k'.format(nb_cnn, bn, noise, nb_train, sig_rate, batch_size, lr, optimizer, num_steps/1000))
generate_folder(direct)
direct_st = direct+'/statistics'
generate_folder(direct_st)

x = tf.placeholder("float", shape=[None, 109,109, 1])
y_ = tf.placeholder("float", shape=[None, 1])

scope_name = 'base'
conv_net, h, pred_logit = conv_classifier(x, nb_cnn = nb_cnn, fc_layers = [128,1],  bn = bn, scope_name = scope_name)

vars_list = tf.trainable_variables(scope_name)
key_list = [v.name[:-2] for v in tf.trainable_variables(scope_name)]
key_var_direct = {}
for key, var in zip(key_list, vars_list):
	key_var_direct[key] = var
saver = tf.train.Saver(key_var_direct, max_to_keep=num_steps)
tf.global_variables()

cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = y_, logits = pred_logit))

# Optimizer
if optimizer=="Adam":
  train_op = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
else:
  train_op= tf.train.GradientDescentOptimizer(lr).minimize(cross_entropy)

## network training
train_loss = []
train_auc = []

test_loss = []
test_auc = []

val_loss = []
val_auc = []
best_val_auc = 0.0

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for i_batch in range(num_steps):
		# generate a batch
		ii = int(i_batch%(nb_train*2/batch_size))
		if ii ==0:
			shuff = np.random.permutation(nb_train*2)
		shuff_batch = shuff[ii*batch_size:(1+ii)*batch_size]
		batch_x = X_trn[shuff_batch,:]
		batch_y = y_trn[shuff_batch].reshape(-1,1)
		batch_x = np.random.normal(batch_x, noise)
		batch_x = (batch_x - np.min(batch_x))/(np.max(batch_x)-np.min(batch_x))
		batch_x = np.expand_dims(batch_x, axis = 3)
		# train the model on a batch
		sess.run(train_op, feed_dict={x: batch_x, y_: batch_y})
		if i_batch % 100 == 0:
			print('>>>>>>The {}-th batch >>>'.format(i_batch))
			train_logit = pred_logit.eval(session=sess,feed_dict={x:batch_x})
			train_loss = np.append(train_loss, cross_entropy.eval(session=sess, feed_dict={x:batch_x, y_:batch_y}))
			train_stat = np.exp(train_logit)
			train_auc = np.append(train_auc,roc_auc_score(batch_y, train_stat))

			test_logit = pred_logit.eval(session=sess,feed_dict={x:X_tst})
			test_loss = np.append(test_loss,cross_entropy.eval(session=sess, feed_dict={x:X_tst, y_:y_tst}))
			test_stat = np.exp(test_logit)
			test_auc = np.append(test_auc,roc_auc_score(y_tst, test_stat))

			val_logit = pred_logit.eval(session=sess,feed_dict={x:X_val})
			val_loss = np.append(val_loss,cross_entropy.eval(session=sess,feed_dict={x:X_tst, y_:y_val}))
			val_stat = np.exp(val_logit)
			val_auc = np.append(val_auc, roc_auc_score(y_val, val_stat))

			print_green('AUC: train {0:0.4f}, val {1:.4f}, test {2:.4f}; loss: train {3:.4f}, val {4:.4f}, test {5:.4f}'.format(train_auc[-1],
				val_auc[-1], test_auc[-1], train_loss[-1], val_loss[-1], test_loss[-1]))
							
			# save the model and results
			model_folder = os.path.join(model_root_folder,os.path.basename(direct))
			generate_folder(model_folder)
			generate_folder(model_folder)
			saver.save(sess, model_folder+'/model', global_step=i_batch)
			print(model_folder)
			# save and plot results
			np.savetxt(direct+'/training_auc.txt',train_auc)
			np.savetxt(direct+'/testing_auc.txt',test_auc)
			np.savetxt(direct+'/training_loss.txt',train_loss)
			np.savetxt(direct+'/testing_loss.txt',test_loss)
			np.savetxt(direct+'/val_loss.txt',val_loss)
			np.savetxt(direct+'/val_auc.txt',val_auc)
			np.savetxt(direct_st+'/statistics_'+str(i_batch)+'.txt',test_stat)
			file_name = os.path.join(direct, 'AUC_over_Iterations_{}.png'.format(os.path.basename(direct)))
			plot_AUCs(file_name, train_auc, val_auc, test_auc)
			# update the best model
			if best_val_auc < val_auc[-1]:
				best_val_auc = val_auc[-1]
				saver.save(sess, model_folder+'/source-best')
				print_red('Update best:'+model_folder)