import os
import glob
import numpy as np
from sklearn.metrics import roc_auc_score

# def load_MRI_true_data(docker = False, train = 65000, val = 600, normal = 1000, anomaly = 1000):
# 	if docker:
# 		dataset_folder = '/data/datasets/MRI'
# 	else:
# 		dataset_folder = '/shared/planck/CommonData/MRI/anomaly_detection_data'
# 	img = np.load(os.path.join(dataset_folder, 'axial_batch2_256x256.npy'))
# 	img_test = img[len(img)-1000:,:]
# 	img_MP =  np.load(os.path.join(dataset_folder, 'axial_batch2_256x256_artifact_1000.npy'))
# 	X_trn, X_val, X_n, X_a = img[:train,:], img[65000:65000+val,:], img_test[:normal,:], img_MP[:anomaly,:]
# 	
# 	return X_trn, X_val, X_n, X_a

# def load_MRI_true_data(docker = False, train = 65000, val = 600, normal = 1000, anomaly = 1000, noise = 0):
# 	if docker:
# 		dataset_folder = '/data/datasets/MRI'
# 	else:
# 		dataset_folder = '/shared/planck/CommonData/MRI/anomaly_detection_data'
# 	img = np.load(os.path.join(dataset_folder, 'axial_batch2_256x256.npy'))
# 	img_MP =  np.load(os.path.join(dataset_folder, 'axial_batch2_256x256_artifact.npy'))
# 	if noise > 0:
# 		gauss1 = np.random.RandomState(0).normal(0, noise, img.shape)
# 		gauss2 = np.random.RandomState(1).normal(0, noise, img_MP.shape)
# 		img, img_MP = img + gauss1, img_MP + gauss2
# 	X_trn, X_val, X_n, X_a = img[:train,:], img[65000:65000+val,:], img[65600:65600+normal,:], img_MP[65600:65600+anomaly,:]
# 	return X_trn, X_val, X_n, X_a

def load_MRI_data(docker = False, train = 65000, val = 600, normal = 1000, anomaly = 1000, noise_level = 0, us_factor = 4):
	if docker:
		dataset_folder = '/data/datasets/MRI'
	else:
		dataset_folder = '/shared/planck/CommonData/MRI/anomaly_detection_data'
	if us_factor == 1 or us_factor ==2 or us_factor == 4:
		us_factor = int(us_factor)
	if noise_level == 0:
		img = np.load(os.path.join(dataset_folder, 'axial_batch2_256x256.npy'))
	else:
		img = np.load(os.path.join(dataset_folder, 'axial_batch2_256x256_artifact-{}_noise-{}.npy'.format(1, noise_level)))
	if noise_level == 0 and us_factor == 4:
		img_MP =  np.load(os.path.join(dataset_folder, 'axial_batch2_256x256_artifact.npy'))
	elif noise_level == 10 and us_factor == 4:
		img_MP =  np.load(os.path.join(dataset_folder, 'axial_batch2_256x256_artifact_noisy.npy'))
	else:
		img_MP =  np.load(os.path.join(dataset_folder, 'axial_batch2_256x256_artifact-{}_noise-{}.npy'.format(us_factor, noise_level)))

	X_trn, X_val, X_n, X_a = img[:train,:], img[65000:65000+val,:], img[65600:65600+normal,:], img_MP[65600:65600+anomaly,:]
	return X_trn, X_val, X_n, X_a

def load_MRI_anomaly(docker = False, train = 65000, val = 200, normal = 1000, anomaly = 1000, noise_level = 0, us_factor = 4):
	if docker:
		dataset_folder = '/data/datasets/MRI'
	else:
		dataset_folder = '/shared/planck/CommonData/MRI/anomaly_detection_data'
	img = np.load(os.path.join(dataset_folder, 'axial_batch2_256x256.npy'))
	print('Loaded shape: {}'.format(img.shape))
	img_MP =  np.load(os.path.join(dataset_folder, 'axial_batch2_256x256_test_null_mask_2x_1000.npy'))
	X_trn, X_val, X_n, X_a = img[:train,:], img[65000:65000+val,:], img[-1000:,:], img_MP
	if False:
		plot_image_pair(dataset_folder+'/image_f_meas_null.png', X_n, X_a, [8,5])
	return X_trn, X_val, X_n, X_a


def load_MRI_true_data(docker = False, train = 65000, val = 600, normal = 1000, anomaly = 1000, noise = 0):
	if docker:
		dataset_folder = '/data/datasets/MRI'
	else:
		dataset_folder = '/shared/planck/CommonData/MRI/anomaly_detection_data'
	img = np.load(os.path.join(dataset_folder, 'axial_batch2_256x256.npy'))
	if noise == 0:
		img_MP =  np.load(os.path.join(dataset_folder, 'axial_batch2_256x256_artifact.npy'))
	else:
		gauss1 = np.random.RandomState(0).normal(0, noise, img.shape)
		img = img + gauss1
		img_MP =  np.load(os.path.join(dataset_folder, 'axial_batch2_256x256_artifact_noisy.npy'))
	X_trn, X_val, X_n, X_a = img[:train,:], img[65000:65000+val,:], img[65600:65600+normal,:], img_MP[65600:65600+anomaly,:]
	return X_trn, X_val, X_n, X_a

def load_MRI_true_Poisson(docker = False, train = 65000, val = 600, normal = 1000, anomaly = 1000, noise = 0):
	if docker:
		dataset_folder = '/data/datasets/MRI'
	else:
		dataset_folder = '/shared/planck/CommonData/MRI/anomaly_detection_data'
	img = np.load(os.path.join(dataset_folder, 'axial_batch2_256x256.npy'))
	img_MP =  np.load(os.path.join(dataset_folder, 'axial_batch2_256x256_artifact.npy'))
	if noise > 0:
# 		img = np.random.RandomState(0).normal(img, noise)
# 		img_MP = np.random.RandomState(1).normal(img_MP, noise)
		pois1 = np.random.RandomState(0).poisson(np.abs(img)).astype(float)
		pois2 = np.random.RandomState(0).poisson(np.abs(img_MP)).astype(float)
		img = img + noise*pois1
		img_MP = img_MP + noise*pois2
	X_trn, X_val, X_n, X_a = img[:train,:], img[65000:65000+val,:], img[65600:65600+normal,:], img_MP[65600:65600+anomaly,:]
	return X_trn, X_val, X_n, X_a