import os
import glob
import numpy as np
from sklearn.metrics import roc_auc_score

def load_MRI_true_data(docker = False, train = 65000, val = 600, normal = 1000, anomaly = 1000):
	if docker:
		dataset_folder = '/data/datasets/MRI'
	else:
		dataset_folder = '/shared/planck/CommonData/MRI/anomaly_detection_data'
	img = np.load(os.path.join(dataset_folder, 'axial_batch2_256x256.npy'))
	img_MP =  np.load(os.path.join(dataset_folder, 'axial_batch2_256x256_artifact_1000.npy'))
	X_trn, X_val, X_n, X_a = img[:train,:], img[65000:65000+val,:], img[65600:65600+normal,:], img_MP[:anomaly,:]
	
	return X_trn, X_val, X_n, X_a