import numpy as np
from sklearn.metrics import roc_auc_score

from load_data import *

def normalize_0_1(data):
	data = np.squeeze(data)
	shp = data.shape
	_shp = (shp[0],)
	for i in range(1,len(shp)):
		_shp = _shp + (1,)
	data = (data - np.amin(np.amin(data, axis = -1), axis = -1).reshape(_shp))/\
			(np.amax(np.amax(data, axis = -1), axis = -1).reshape(_shp)-\
			np.amin(np.amin(data, axis = -1), axis = -1).reshape(_shp))
	image_sum = np.squeeze(np.apply_over_axes(np.sum, data, axes = [1,2]))
	return data[~np.isnan(image_sum),:]

# visualize one group of examples
def save_recon_images_1(img_file_name, imgs, recons, fig_size):
	from matplotlib.backends.backend_agg import FigureCanvasAgg
	from matplotlib.figure import Figure
	imgs, recons = np.squeeze(imgs), np.squeeze(recons)
	test_size = imgs.shape[0]
	indxs = np.random.randint(0,int(test_size),3)
# 	fig_size = (8,6)
	fig_size = fig_size
	fig = Figure(figsize=fig_size)
	rows, cols = 2, 3
	ax = fig.add_subplot(rows, cols, 1); cax=ax.imshow(imgs[indxs[0],:],cmap='gray'); fig.colorbar(cax); ax.set_title('Image-{}'.format(indxs[0])); ax.set_ylabel('f') 
	ax = fig.add_subplot(rows, cols, 2); cax=ax.imshow(imgs[indxs[1],:],cmap='gray'); fig.colorbar(cax); ax.set_title('Image-{}'.format(indxs[1]));
	ax = fig.add_subplot(rows, cols, 3); cax=ax.imshow(imgs[indxs[2],:],cmap='gray'); fig.colorbar(cax); ax.set_title('Image-{}'.format(indxs[2]));
	ax = fig.add_subplot(rows, cols, 4); cax=ax.imshow(recons[indxs[0],:],cmap='gray'); fig.colorbar(cax); ax.set_ylabel('f_MP')
	ax = fig.add_subplot(rows, cols, 5); cax=ax.imshow(recons[indxs[1],:],cmap='gray'); fig.colorbar(cax);
	ax = fig.add_subplot(rows, cols, 6); cax=ax.imshow(recons[indxs[2],:],cmap='gray'); fig.colorbar(cax);
	canvas = FigureCanvasAgg(fig)
	canvas.print_figure(img_file_name, dpi=100)

def plot_hist(file_name, x, y):
	import matplotlib.pyplot as plt
	from matplotlib.backends.backend_agg import FigureCanvasAgg
	from matplotlib.figure import Figure
	kwargs = dict(alpha=0.6, bins=100, density= True, stacked=True)
	fig_size = (8,6)
	fig = Figure(figsize=fig_size)
	file_name = file_name
	ax = fig.add_subplot(111)
# 	x = np.exp(-x)
# 	y = np.exp(-y)
	ax.hist(x, **kwargs, color='g', label='Norm')
	ax.hist(y, **kwargs, color='r', label='Anomaly')
	title = os.path.basename(os.path.dirname(file_name))
	ax.set_title(title)
	ax.set_xlabel('Error')
	ax.set_ylabel('Probalbity')
	ax.legend(['Norm', 'Anomaly'])
	ax.set_xlim([np.min(np.concatenate([x,y])), np.max(np.concatenate([x,y]))])
	canvas = FigureCanvasAgg(fig)
	canvas.print_figure(file_name, dpi=100)

_, _, X_SA_tst, X_SP_tst = load_MRI_true_data(docker = True, train = 1000, val = 200, normal = 200, anomaly = 200, noise = 0)
noise = 0
pois1 = np.random.RandomState(0).poisson(np.abs(X_SA_tst)).astype(float)
pois2 = np.random.RandomState(0).poisson(np.abs(X_SP_tst)).astype(float)
X_SA_pois = X_SA_tst + noise*pois1
X_SP_pois = X_SP_tst + noise*pois2
X_SA_pois, X_SP_pois = normalize_0_1(X_SA_pois), normalize_0_1(X_SP_pois)
Xt = np.concatenate([X_SA_pois, X_SP_pois], axis = 0)
yt = np.concatenate([np.zeros((len(X_SA_pois),1)), np.ones((len(X_SP_pois),1))], axis = 0).flatten()
# Xt_n = Xt + pois
img_means = np.squeeze(np.apply_over_axes(np.mean, Xt, axes = [1,2]))
mean_auc = roc_auc_score(yt, img_means)
save_recon_images_1('/data/datasets/MRI/poisson_noisy_image_{}.png'.format(noise), X_SA_pois, X_SP_pois, [11, 5])
print('The AUC based on pixel mean: {0:.4f}'.format(mean_auc))

x = np.squeeze(np.apply_over_axes(np.mean, X_SA_pois, axes = [1,2]))
y = np.squeeze(np.apply_over_axes(np.mean, X_SP_pois, axes = [1,2]))
plot_hist('/data/datasets/MRI/poisson_noisy_hist_{}.png'.format(noise), x, y)