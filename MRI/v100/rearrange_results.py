import os
import glob

root_folder = '/scratch1/fs1/anastasio/Data_FDA_Breast/DA_Observer'
result_root_folder = os.path.join(root_folder, 'results')
new_root_folder = os.path.join(root_folder, 'backup')

def generate_folder(folder):
# 	print(os.path.exists(folder))
	if not os.path.exists(folder):
		os.system('mkdir {}'.format(folder))

generate_folder(new_root_folder)
# sub_folder_list = ['CLB', 'CLB-FDA', 'DA-TF', 'FDA']
sub_folder_list = ['CLB-FDA']
result_folders = glob.glob(result_root_folder+'/*')
for folder in result_folders:
	print('Go into {}'.format(os.path.relpath(folder, root_folder)))
	if os.path.isdir(folder) and os.path.basename(folder) in sub_folder_list:
		generate_folder(os.path.join(new_root_folder, os.path.basename(folder)))
		base_folders = glob.glob(os.path.join(folder, '*'))
		for base_folder in base_folders:
			if os.path.isdir(base_folder):
				print('Go into {}'.format(os.path.relpath(base_folder, root_folder)))
				new_base_folder = base_folder.replace(result_root_folder, new_root_folder)
				generate_folder(new_base_folder)
				model_folders = glob.glob(os.path.join(base_folder, '*'))
				for model_folder in model_folders:
					if os.path.isdir(model_folder) and not 'statistics' in os.path.basename(model_folder):
						print('Go into {}'.format(os.path.relpath(model_folder, root_folder)))
						## copy the models and results
						new_model_folder = model_folder.replace(result_root_folder, new_root_folder)
						generate_folder(new_model_folder)
						os.system('cp -f {}/target_best* {}/*.png {}/*.txt {}'.format(model_folder, model_folder, model_folder, new_model_folder))
					elif model_folder.endswith('.png') or model_folder.endswith('.txt') or 'source-best' in model_folder:
						os.system('cp -f {} {}'.format(model_folder, new_base_folder))