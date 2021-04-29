

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from .path_tools import *

def sort_kernel(kernel, allocation, item_labels=[] ):
	alloc_list = np.unique(allocation)
	n = kernel.shape[0]
	rearrangement = []
	cluster_by_id = {}
	cluster_by_name = {}
	sorted_kernel = np.empty((0, n))

	for m in alloc_list:
		new_list = np.where(allocation == m)[0].tolist()
		cluster_by_id[m] = new_list
		rearrangement.extend(new_list)

		if len(item_labels) > 0:
			cluster_by_name[m] = []
			for q in new_list:
				cluster_by_name[m].append( item_labels[q] )


		f = kernel[allocation == m, :]
		sorted_kernel = np.vstack((sorted_kernel, f))
		
	H_sorted_kernel = np.empty((n,0))
	for m in alloc_list:
		f = sorted_kernel[:,allocation == m]
		H_sorted_kernel = np.hstack((H_sorted_kernel, f))

	sorted_kernel = H_sorted_kernel


	#	Sort the labels if they exist
	if len(item_labels) > 0:
		sorted_labels = []
		for m in rearrangement:
			sorted_labels.append( item_labels[m] )

	return H_sorted_kernel

def plot_heatMap(db, K, layer_id):
	K = sort_kernel(K, db['Y'])
	data_name = db['data_name'].split('/')[1]


	plt.imshow(1-K, cmap='Blues_r', interpolation='nearest') #cmap options = viridis,Blues_r,hot
	plt.title('%s : Kernel at Layer %d'%(data_name , layer_id), fontsize=17, fontweight='bold' )
	plt.xlabel('Sample ID', fontsize=13, fontweight='bold' )
	plt.ylabel('Sample ID', fontsize=13, fontweight='bold' )
	#plt.show()

	pth = './results/' + db['data_name']
	ensure_path_exists(pth)
	plt.savefig(pth + '/kernel_' + str(layer_id) + '.png')
	plt.close()



def plot_目(db, score_目, Ł_目, title, xlabel, n, ylabel=''):
	plt.clf()
	目 = zip(score_目, Ł_目)

	tmp_score_目 = []
	for l in score_目:
		tmp_score_目.append(l[0])
	tmp_score_目 = np.array(tmp_score_目)
	hpos = np.ones(tmp_score_目.shape)

	ℓ = len(tmp_score_目)
	score_sort_args = np.argsort(tmp_score_目, axis=0)
	hpos[score_sort_args] = (1.0/(ℓ) )*np.arange(1, ℓ+1)
	Δ = tmp_score_目 - hpos

#	print(tmp_score_目, '\n')
#	print(score_sort_args, '\n')
#	print(hpos, '\n')
#	print(Δ, '\n')
#	import pdb; pdb.set_trace()	

	idx = 0
	目 = zip(score_目, Ł_目)
	for score, Ł in 目:
		plt.plot(score)
		#plt.arrow(n+1.8, hpos[idx], -1.8, Δ[idx], head_width=0.05, head_length=0.080, linestyle='dotted', fc='k', ec='k')
		#plt.arrow(0, score[0], 2.0, -Δ[idx], head_width=0.03, head_length=0.080, linestyle=':', fc='k', ec='k')
		#cir = plt.Circle((0, score[0]), 0.01)
		#plt.gcf().gca().add_artist(cir)
		plt.text(-0.1,score[0], Ł, ha='right', va='center', fontsize=10, fontweight='bold' )
		plt.axis([-0.6, len(score)-1, -0.1, 1.05])
		idx += 1

	#import pdb; pdb.set_trace()
	plt.xlabel(xlabel, fontsize=15, fontweight='bold' )
	plt.ylabel(ylabel, fontsize=15, fontweight='bold' )
	plt.title(title, fontsize=18, fontweight='bold' )
	plt.xticks(np.arange(len(score_目[0])), np.arange(1, len(score_目[0])+1))
	#plt.show()

	pth = './results/' + db['data_name']
	ensure_path_exists(pth)
	plt.savefig(pth + '/score_layer_progression.png')
	plt.close()


def plot_3D(db, X, Y, layer_id):
	clist = ['r','g','b','c','y','k']

	fig = plt.figure()
	ax = fig.gca(projection='3d')

	labels = np.unique(Y)
	for l in labels:
		subX = X[Y == l, :]
		ax.scatter(subX[:,0], subX[:,1], np.zeros((len(subX[:,1]), 1)), c=clist[l], marker='o')

	#ax.scatter(X[:,0], X[:,1], X[:,2], c='b', marker='o')

	ax.set_title('Scatter plot after %d layers'%layer_id)
	pth = './results/' + db['data_name']
	plt.savefig(pth + '/Scatter_3d_layer%d.png'%layer_id)
	plt.close()

def scatter(db, X, Y, layer_id):
	clist = ['r','g','b','c','y','k']

	labels = np.unique(Y)
	for l in labels:
		subX = X[Y == l, :]
		plt.scatter(subX[:,0], subX[:,1], c=clist[l], marker='o')

	plt.title('Scatter plot after %d layers'%layer_id)
	plt.tight_layout()
	pth = './results/' + db['data_name']
	plt.savefig(pth + '/Scatter_3d_layer%d.png'%layer_id)
	plt.close()

