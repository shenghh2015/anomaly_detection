import os
import numpy as np
import glob

def write_fit_roc(t0, t1, fit_input_file):
# 	fit_input_file = result_folder + '/roc_fit_input.txt'
	if os.path.exists(fit_input_file):
		os.system('rm -f {}'.format(fit_input_file))
	with open(fit_input_file, 'w+') as f:
		f.write('LABROC\n')
		f.write('Large\n')
		for i in range(len(t0)):
			f.write('{0:.6f}\n'.format(t0[i]))
		f.write('*\n')
		for i in range(len(t1)):
			f.write('{0:.6f}\n'.format(t1[i]))
		f.write('*\n')

result_root_folder = 'experiments/MRI_AE'
result_folders = glob.glob(result_root_folder+'/*')

for result_folder in result_folders:
	if os.path.isdir(result_folder):
		if os.path.exists(result_folder+'/AE_stat.txt'):
			stat_arr = np.loadtxt(result_folder+'/AE_stat.txt')
			pixel_mean_arr = np.loadtxt(result_folder+'/Pixel_mean_stat.txt')
		else:
			continue
		## write the results for AE
		if len(stat_arr) == 0:
			continue
		stat_arr = stat_arr.flatten()
		stat_len = len(stat_arr)
		t0 = stat_arr[:int(stat_len/2)]
		t1 = stat_arr[int(stat_len/2):]
		if not len(t0) == len(t1):
			continue
		write_fit_roc(t0, t1, result_folder+'/AE_roc_fit_input.txt')
		## write the results for pixel mean
		if len(pixel_mean_arr)==0:
			continue
		stat_arr = pixel_mean_arr.flatten()
		stat_len = len(pixel_mean_arr)
		t0 = pixel_mean_arr[:int(stat_len/2)]
		t1 = pixel_mean_arr[int(stat_len/2):]
		if not len(t0) == len(t1):
			continue
		write_fit_roc(t0, t1, result_folder+'/pixel_roc_fit_input.txt')
		# write roc fit input file
		