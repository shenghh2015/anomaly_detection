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
def load_anomaly_data(docker = True, dataset = 'total', train = 80000, valid = 400, test = 400):
	if docker:
		dataset_folder = '/data/datasets'
	else:
		dataset_folder = '/shared/planck/Phantom/Breast_Xray'
	X_SA = np.load('{}/FDA_DM_ROIs/npy_dataset/{}_SA.npy'.format(dataset_folder, dataset))
	X_SP = np.load('{}/FDA_DM_ROIs/npy_dataset/{}_SP.npy'.format(dataset_folder, dataset))
	if dataset == 'dense':
		offset_valid = 7100
	elif dataset == 'hetero':
		offset_valid = 36000
	elif dataset == 'scattered':
		offset_valid = 33000
	elif dataset == 'fatty':
		offset_valid = 9000
	elif dataset == 'total':
		offset_valid = 85000
	offset_test = 400 + offset_valid
	X_SA_trn, X_SA_val, X_SA_tst = X_SA[:train,:], X_SA[offset_valid:offset_valid+valid,:], X_SA[offset_test:offset_test+test,:]
	_, _, X_SP_tst = X_SP[:train,:], X_SP[offset_valid:offset_valid+valid,:], X_SP[offset_test:offset_test+test,:]
	print('---- Anomaly detection Dataset Summary: {} ----'.format(dataset))
	print(' -train normal {}'.format(X_SA_trn.shape[0]))
	print(' -valid normal {}'.format(X_SA_val.shape[0]))
	print(' -test SA {}, SP {}'.format(X_SA_tst.shape[0], X_SP_tst.shape[0]))
	return X_SA_trn, X_SA_val, X_SA_tst, X_SP_tst

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