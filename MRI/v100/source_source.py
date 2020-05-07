import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from natsort import natsorted
		
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


dataset = '/data/results/CLB'
# model_name = 'noise-5.0-train-100000-sig-0.035'

model_folders = glob.glob(dataset+'/*')

for model_folder in model_folders:
	if os.path.isdir(model_folder) and os.path.exists(model_folder+'/model-0.meta'):
		model_name = os.path.basename(model_folder)
		model_list = glob.glob(os.path.join(dataset, model_name, 'model-*.meta'))
		train_AUCs = np.loadtxt(os.path.join(dataset, model_name, 'training_auc.txt'))
		val_AUCs = np.loadtxt(os.path.join(dataset, model_name, 'val_auc.txt'))
		test_AUCs = np.loadtxt(os.path.join(dataset, model_name, 'testing_auc.txt'))

		# check the best
		iter_best = np.argmax(val_AUCs)
		train_AUC = train_AUCs[np.argmax(val_AUCs)]
		val_AUC_best = val_AUCs[np.argmax(val_AUCs)]
		test_AUC = test_AUCs[np.argmax(val_AUCs)]
		natsorted(model_list)
		model_best = model_list[iter_best]

		# mark the best model
		import re
		splits = re.split('\-|\.',os.path.basename(model_best))
		step_num = splits[1]
		meta_best_old = os.path.join(os.path.dirname(model_best), '{}-{}.meta'.format(splits[0], step_num))
		index_best_old = os.path.join(os.path.dirname(model_best), '{}-{}.index'.format(splits[0], step_num))
		data_best_old = os.path.join(os.path.dirname(model_best), '{}-{}.data-00000-of-00001'.format(splits[0], step_num))
		meta_best_new = os.path.join(os.path.dirname(model_best), '{}-{}.meta'.format('source', 'best'))
		index_best_new = os.path.join(os.path.dirname(model_best), '{}-{}.index'.format('source', 'best'))
		data_best_new = os.path.join(os.path.dirname(model_best), '{}-{}.data-00000-of-00001'.format('source', 'best'))
		os.system('cp -f {} {}'.format(index_best_old, index_best_new))
		os.system('cp -f {} {}'.format(meta_best_old, meta_best_new))
		os.system('cp -f {} {}'.format(data_best_old, data_best_new))

		# print out results
		print('>>>>>>> {} <<<<<<<'.format(model_name))
		print('AUC: train {0:.4f}, val {1:.4f}, test {2:.4f}'.format(train_AUC, val_AUC_best, test_AUC))
		print('best model: {}'.format(os.path.basename(model_best)))
		print('\n')
		# plot train, valid, test curves
		file_name = os.path.join(dataset, model_name, 'AUC_over_Iterations_{}.png'.format(model_name))
		plot_AUCs(file_name, train_AUCs, val_AUCs, test_AUCs)


