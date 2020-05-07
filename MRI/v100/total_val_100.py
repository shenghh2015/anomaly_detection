import tensorflow as tf
import os
import numpy as np
import glob
from natsort import natsorted
from termcolor import colored 
import argparse

from load_data import *
from model import *

def str2bool(value):
    return value.lower() == 'true'

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

## load weights and calculate the AUC and test statistics
def run_evaluation(model_list, model_set):
	np.random.shuffle(model_list)   ## raondom shuffle it, and allow to parallely run the jobs
	for model_name in model_list:
		# model_set = 'CLB-FDA'
		# model_name = 'TF-lr-1e-06-bz-50-iter-50000-scr-False-fc-128-bn-False-trg_labels-70-clf_v1-total'
		print_yellow(model_name)
		if model_set == 'FDA':
			model_meta_files = glob.glob(os.path.join(output_folder, model_set, model_name, 'model*.meta'))
			best_model_meta = os.path.join(output_folder, model_set, model_name, 'target-best.meta')
			new_best_meta = os.path.join(output_folder, model_set, model_name, 'val_100_target_best.meta')
		else:
			model_meta_files = glob.glob(os.path.join(output_folder, model_set, base_model, model_name, 'target-*.meta'))
			best_model_meta = os.path.join(output_folder, model_set, base_model, model_name, 'target_best.meta')
			new_best_meta = os.path.join(output_folder, model_set, base_model, model_name, 'val_100_target_best.meta')
		if os.path.exists(new_best_meta):
			continue
		model_meta_files = natsorted(model_meta_files)
		if os.path.exists(best_model_meta):
			model_meta_files.insert(0, best_model_meta)
		print_yellow('Amount of models:{}'.format(len(model_meta_files)))
		best_val_auc = 0
		select_test_auc = 0
		val_auc_list = []
		test_auc_list = []
		with tf.Session() as sess:
			if len(model_meta_files) > 2:
				for model_meta in model_meta_files:
					tf.global_variables_initializer().run(session=sess)
					target_saver.restore(sess, model_meta.replace('.meta', ''))
					# AUC calculation
					test_target_logit = target_logit.eval(session=sess,feed_dict={xt:Xt_tst})
					test_target_stat = np.exp(test_target_logit)
					test_target_AUC = roc_auc_score(yt_tst, test_target_stat)
					val_target_logit = target_logit.eval(session=sess,feed_dict={xt:Xt_val})
					val_target_stat = np.exp(val_target_logit)
					val_target_AUC = roc_auc_score(yt_val, val_target_stat)
					if not model_meta == best_model_meta:
						test_auc_list.append(test_target_AUC)
						val_auc_list.append(val_target_AUC)
						np.savetxt(os.path.join(os.path.dirname(model_meta),'test_auc_100.txt'), test_auc_list)
						np.savetxt(os.path.join(os.path.dirname(model_meta),'val_auc_100.txt'), val_auc_list)
					if best_val_auc < val_target_AUC:
						best_val_auc = val_target_AUC
						select_test_auc = test_target_AUC
						select_model_meta = model_meta
		# 				target_saver.save(sess, os.path.dirname(model_meta)+'/val_100_target_best')
						print_red('Update best based on val 100:'+os.path.dirname(model_meta))
						print_green('Update: AUC: T-test {0:.4f}, T-valid {1:.4f}'.format(select_test_auc, best_val_auc))
# 					print('AUC: T-test {0:.4f}, T-valid {1:.4f}'.format(test_target_AUC, val_target_AUC))
				# calculate the statistics for the selected model
				tf.global_variables_initializer().run(session=sess)
				target_saver.restore(sess, select_model_meta.replace('.meta', ''))
				# AUC calculation
				test_target_logit = target_logit.eval(session=sess,feed_dict={xt:Xt_tst})
				test_target_stat = np.exp(test_target_logit)
				test_target_AUC = roc_auc_score(yt_tst, test_target_stat)
				val_target_logit = target_logit.eval(session=sess,feed_dict={xt:Xt_val})
				val_target_stat = np.exp(val_target_logit)
				val_target_AUC = roc_auc_score(yt_val, val_target_stat)
				# save the selected statistics
				np.savetxt(os.path.join(os.path.dirname(select_model_meta),'best_test_stat_100.txt'), test_target_stat)
				np.savetxt(os.path.join(os.path.dirname(select_model_meta),'best_test_auc_100.txt'), [test_target_AUC])
				np.savetxt(os.path.join(os.path.dirname(select_model_meta),'best_val_auc_100.txt'), [val_target_AUC])
				target_saver.save(sess, os.path.dirname(select_model_meta)+'/val_100_target_best')
				print_green('AUC: T-test {0:.4f}, T-valid {1:.4f}'.format(test_target_AUC, val_target_AUC))
				plot_auc_iterations(test_auc_list, val_auc_list, os.path.dirname(model_meta)+'/val_100_AUC_{}.png'.format(os.path.basename(os.path.dirname(select_model_meta))))

TF_list = ['TF-lr-1e-06-bz-50-iter-50000-scr-False-fc-128-bn-False-trg_labels-70-clf_v1-total',
			'TF-lr-5e-06-bz-400-iter-50000-scr-False-fc-128-bn-False-trg_labels-100-clf_v1-total',
			'TF-lr-1e-06-bz-100-iter-50000-scr-False-fc-128-bn-False-trg_labels-200-clf_v1-total',
			'TF-lr-5e-06-bz-100-iter-50000-scr-False-fc-128-bn-False-trg_labels-200-clf_v1-total',
			'TF-lr-5e-06-bz-100-iter-50000-scr-False-fc-128-bn-False-trg_labels-300-clf_v1-total',
			'TF-lr-1e-06-bz-100-iter-20000-scr-False-fc-128-bn-False-trg_labels-400',
			'TF-lr-1e-06-bz-100-iter-10000-scr-False-fc-128-bn-False-trg_labels-500']
mmd_list = ['mmd-1.0-lr-1e-05-bz-400-iter-100000-scr-None-shar-True-fc-128-bn-False',
		   'mmd-1.0-lr-1e-05-bz-400-iter-100000-scr-None-shar-True-fc-128-bn-False',
		   'mmd-1.0-lr-1e-05-bz-400-iter-50000-scr-False-shar-True-fc-128-bn-False-tclf-0.1-sclf-1.0-tlabels-70-vclf-1-total',
		   'mmd-1.0-lr-0.0001-bz-400-iter-50000-scr-False-shar-True-fc-128-bn-False-tclf-1.0-sclf-1.0-tlabels-70-vclf-1-total',
		   'mmd-1.0-lr-1e-05-bz-400-iter-50000-scr-False-shar-True-fc-128-bn-False-tclf-1.0-sclf-1.0-trg_labels-100-vclf-1',
		   'mmd-1.0-lr-0.0001-bz-400-iter-50000-scr-False-shar-True-fc-128-bn-False-tclf-1.0-sclf-1.0-trg_labels-200-vclf-1',
		   'mmd-1.0-lr-1e-05-bz-400-iter-50000-scr-False-shar-True-fc-128-bn-False-tclf-1.0-sclf-1.0-trg_labels-300-vclf-1',
		   'mmd-1.0-lr-0.0001-bz-400-iter-50000-scr-False-shar-True-fc-128-bn-False-tclf-1.0-sclf-1.0-trg_labels-300-vclf-1',
		   'mmd-1.0-lr-1e-05-bz-400-iter-50000-scr-False-shar-True-fc-128-bn-False-tclf-1.0-sclf-1.0-trg_labels-400-vclf-1',
		   'mmd-1.0-lr-1e-05-bz-400-iter-100000-scr-True-shar-True-fc-128-bn-False-tclf-1.0-sclf-1.0-trg_labels-500']
naive_list = ['cnn-4-bn-False-trn-70-bz-50-lr-1e-05-Adam-stp-25.0k-total',
			  'cnn-4-bn-False-trn-100-bz-100-lr-1e-05-Adam-stp-25.0k-total',
			  'cnn-4-bn-False-trn-200-bz-100-lr-1e-05-Adam-stp-25.0k-total',
			  'cnn-4-bn-False-trn-300-bz-100-lr-1e-05-Adam-stp-25.0k-total',
			  'cnn-4-bn-False-trn-400-bz-100-lr-1e-05-Adam-stp-25.0k-total',
			  'cnn-4-bn-False-trn-500-bz-400-lr-1e-05-Adam-stp-20.0k']

parser = argparse.ArgumentParser()
parser.add_argument("gpu", type=int, default = 0)
parser.add_argument("docker", type = str2bool, default = True)

args = parser.parse_args()
print(args)

gpu = args.gpu
docker = args.docker
# docker = True
# gpu = 1

if docker:
	output_folder = '/data/results'
else:
	output_folder = './data'

base_model = 'cnn-4-bn-False-noise-2.0-trn-100000-sig-0.035-bz-400-lr-5e-05-Adam-100.0k'
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

## load target data
_, Xt_val, Xt_tst, _, yt_val, yt_tst = load_target(dataset = 'total', train = 1000, valid = 100, test = 400)
Xt_val, Xt_tst = (Xt_val-np.min(Xt_val))/(np.max(Xt_val)-np.min(Xt_val)), (Xt_tst-np.min(Xt_tst))/(np.max(Xt_tst)-np.min(Xt_tst))
Xt_val, Xt_tst = np.expand_dims(Xt_val, axis = 3), np.expand_dims(Xt_tst, axis = 3)

## create target model
xt = tf.placeholder("float", shape=[None, 109,109, 1])
yt = tf.placeholder("float", shape=[None, 1])
_, _, target_logit = conv_classifier(xt, nb_cnn = 4, fc_layers = [128,1],  bn = False, scope_name = 'base')
scope_name = 'base'
vars_list = tf.trainable_variables(scope_name)
key_list = [v.name[:-2] for v in tf.trainable_variables(scope_name)]
key_var_direct = {}
for key, var in zip(key_list, vars_list):
	key_var_direct[key] = var
target_saver = tf.train.Saver(key_var_direct, max_to_keep=1)

# run evaluation
run_evaluation(mmd_list + TF_list, 'CLB-FDA')
# run_evaluation(TF_list, 'CLB-FDA')
run_evaluation(naive_list, 'FDA')