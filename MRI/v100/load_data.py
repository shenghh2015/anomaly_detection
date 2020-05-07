import os
import glob
import numpy as np
from sklearn.metrics import roc_auc_score

dataset_folder = '/data/datasets'
def load_source(train = 80000, valid = 400, test = 400, sig_rate = 0.035):
# 	train = 80000
# 	valid = 400
# 	test = 400
# 	sig_rate = 0.035
# 	sig_file = '/shared/planck/Phantom/Breast_Xray/FDA_signals/hetero_sig.dat'
# 	CLB_file = '/shared/rsaas/shenghua/CLB/CLB_128N_400000IM.npy'
	sig_file = os.path.join(dataset_folder, 'FDA_signals/hetero_sig.dat')
	CLB_file = os.path.join(dataset_folder, 'CLB/CLB_128N_400000IM.npy')
	sig = np.fromfile(sig_file, dtype = np.float32).reshape(109,109)
	data = np.load(CLB_file, mmap_mode='r')
	X = data[:,:,0:train+valid+test]
	X = np.transpose(np.reshape(X, [128, 128, train+valid+test], order='F'))
	X_SA_trn, X_SA_val, X_SA_tst = X[:train, 64-54:64+55,64-54:64+55], X[train:train+valid, 64-54:64+55,64-54:64+55],\
		X[train+valid:train+valid+test, 64-54:64+55,64-54:64+55]
	X_SP_trn, X_SP_val, X_SP_tst = X_SA_trn + sig * sig_rate, X_SA_val + sig * sig_rate, X_SA_tst + sig * sig_rate
	X_trn, X_val, X_tst = np.concatenate([X_SA_trn, X_SP_trn]), np.concatenate([X_SA_val, X_SP_val]), np.concatenate([X_SA_tst, X_SP_tst])
	y_trn = np.concatenate([np.zeros((train,1)), np.ones((train,1))]).flatten()
	y_val = np.concatenate([np.zeros((valid,1)), np.ones((valid,1))]).flatten()
	y_tst = np.concatenate([np.zeros((test,1)), np.ones((test,1))]).flatten()

	return X_trn, X_val, X_tst, y_trn, y_val, y_tst

def load_target(dataset = 'total', train = 80000, valid = 400, test = 400):
# 	dataset = 'total'
# 	train = 80000
# 	valid = 400
# 	test = 400
# 	X_SA = np.load('/shared/planck/Phantom/Breast_Xray/FDA_DM_ROIs/npy_dataset/{}_SA.npy'.format(dataset))
# 	X_SP = np.load('/shared/planck/Phantom/Breast_Xray/FDA_DM_ROIs/npy_dataset/{}_SP.npy'.format(dataset))
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
	X_SA = np.load(os.path.join(dataset_folder, 'FDA_DM_ROIs/npy_dataset/{}_SA.npy'.format(dataset)))
	X_SP = np.load(os.path.join(dataset_folder, 'FDA_DM_ROIs/npy_dataset/{}_SP.npy'.format(dataset)))
	X_SA_trn, X_SA_val, X_SA_tst = X_SA[:train,:], X_SA[offset_valid:offset_valid+valid,:], X_SA[offset_test:offset_test+test,:]
	X_SP_trn, X_SP_val, X_SP_tst = X_SP[:train,:], X_SP[offset_valid:offset_valid+valid,:], X_SP[offset_test:offset_test+test,:]
	X_trn, X_val, X_tst = np.concatenate([X_SA_trn, X_SP_trn]), np.concatenate([X_SA_val, X_SP_val]), np.concatenate([X_SA_tst, X_SP_tst])
	y_trn = np.concatenate([np.zeros((train,1)), np.ones((train,1))]).flatten()
	y_val = np.concatenate([np.zeros((valid,1)), np.ones((valid,1))]).flatten()
	y_tst = np.concatenate([np.zeros((test,1)), np.ones((test,1))]).flatten()
	print('---- Dataset Summary: {} ----'.format(dataset))
	print(' -all SA {}, SP {}'.format(X_SA.shape[0], X_SP.shape[0]))
	print(' -trn SA {}, SP {}'.format(X_SA_trn.shape[0], X_SP_trn.shape[0]))
	print(' -val SA {}, SP {}'.format(X_SA_val.shape[0], X_SP_val.shape[0]))
	print(' -val SA {}, SP {}'.format(X_SA_tst.shape[0], X_SP_tst.shape[0]))
# 	print('\n')
	return X_trn, X_val, X_tst, y_trn, y_val, y_tst

def load_target_archive(dataset = 'total', train = 80000, valid = 400, test = 400):
# 	dataset = 'total'
# 	train = 80000
# 	valid = 400
# 	test = 400
	X_SA = np.fromfile(os.path.join(dataset_folder, 'FDA_DM_ROIs/npy_dataset/{}_SA.dat'.format(dataset)), dtype = np.float32)
	X_SA = X_SA.reshape(-1,109,109)
	shuff = np.random.RandomState(2).permutation(X_SA.shape[0])
	X_SA = X_SA[shuff]
	X_SP = np.fromfile(os.path.join(dataset_folder, 'FDA_DM_ROIs/npy_dataset/{}_SA.dat'.format(dataset)), dtype = np.float32)
	X_SP = X_SP.reshape(-1,109,109)
	shuff = np.random.RandomState(3).permutation(X_SP.shape[0])
	X_SP = X_SP[shuff]
	X_SA_trn, X_SA_val, X_SA_tst = X_SA[:train,:], X_SA[train:train+valid,:], X_SA[train+valid:train+valid+test,:]
	X_SP_trn, X_SP_val, X_SP_tst = X_SP[:train,:], X_SP[train:train+valid,:], X_SP[train+valid:train+valid+test,:]
	X_trn, X_val, X_tst = np.concatenate([X_SA_trn, X_SP_trn]), np.concatenate([X_SA_val, X_SP_val]), np.concatenate([X_SA_tst, X_SP_tst])
	y_trn = np.concatenate([np.zeros((train,1)), np.ones((train,1))]).flatten()
	y_val = np.concatenate([np.zeros((valid,1)), np.ones((valid,1))]).flatten()
	y_tst = np.concatenate([np.zeros((test,1)), np.ones((test,1))]).flatten()
	print('---- Dataset Summary: {} ----'.format(dataset))
	print(' -all SA {}, SP {}'.format(X_SA.shape[0], X_SP.shape[0]))
	print(' -trn SA {}, SP {}'.format(X_SA_trn.shape[0], X_SP_trn.shape[0]))
	print(' -val SA {}, SP {}'.format(X_SA_val.shape[0], X_SP_val.shape[0]))
	print(' -val SA {}, SP {}'.format(X_SA_tst.shape[0], X_SP_tst.shape[0]))
# 	print('\n')
	return X_trn, X_val, X_tst, y_trn, y_val, y_tst

def evaluate_target_HO(dataset = 'total', train = 80000, valid = 400, test = 400):
# 	dataset = 'dense'
# 	train = 7000
# 	valid = 400
# 	test = 400
	X_trn, X_val, X_tst, y_trn, y_val, y_tst = load_target(dataset = dataset, train = train, valid = valid, test = test)
	H0, H1 = X_trn[:train,:], X_trn[train:,:]
	H0, H1 = H0.reshape(H0.shape[0],-1), H1.reshape(H1.shape[0],-1)
	K0, K1 = np.cov(H0.T), np.cov(H1.T)
	K, dg = 0.5*(K0+K1), (np.mean(H1, axis = 0)-np.mean(H0, axis = 0))
	W = np.matmul(np.linalg.inv(K),dg)
	H_tst = X_tst.reshape(X_tst.shape[0],-1)
	scores = np.matmul(H_tst, W)
	auc_score = roc_auc_score(y_tst, scores.flatten())
	print('HO AUC:{0:.3f}'.format(auc_score))
#                 trn_h0 = bk[:100000,:,:]
#                 shp = trn_h0.shape
#                 trn_h0 = np.reshape(trn_h0, (shp[0],-1))
#                 Kb = np.cov(trn_h0.T)
#                 sigma = 20
#                 Kn = sigma**2*np.eye(64*64)
#                 K = Kb + Kn
#                 K_inv = np.linalg.inv(K)
#                 sig_ = sig.flatten()
#                 W = np.matmul(K_inv,sig_)
# 
#                 # plt.ion()
#                 # plt.imshow(W.reshape((64,64))
# 
#                 tst_h0 = bk[100000:100000+100,:,:].reshape((100,64*64))
#                 tst_h0 = np.random.RandomState(12).normal(tst_h0, 20)
#                 tst_h1 = tst_h0 + sig_
#                 tst_h1 = np.random.RandomState(12).normal(tst_h1, 20)
#                 test_data = np.concatenate([tst_h0,tst_h1], axis = 0)
#                 y_scores = np.matmul(test_data, W)
#                 y_true = np.concatenate([np.zeros(100), np.ones(100)])
#                 auc_score = roc_auc_score(y_true, y_scores)

# import matplotlib.pyplot as plt
# plt.close('all')   
# plt.ion()
# fig = plt.figure()
# for i in range(len(X_trn)):
# 	if i > 100:
# 		break
# 	ax = fig.add_subplot(231)
# 	bx = fig.add_subplot(232)
# 	cx = fig.add_subplot(233)
# 	ax1 = fig.add_subplot(234)
# 	bx1 = fig.add_subplot(235)
# 	cx1 = fig.add_subplot(236)
# 	ax.imshow(X_SA_trn[i,:])
# 	bx.imshow(X_SA_val[i,:])
# 	cx.imshow(X_SA_tst[i,:])
# 	ax1.imshow(X_SP_trn[i,:])
# 	bx1.imshow(X_SP_val[i,:])
# 	cx1.imshow(X_SP_tst[i,:])
# 	plt.pause(1)

# def load_target():
