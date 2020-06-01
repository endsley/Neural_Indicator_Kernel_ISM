
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from src.networks.ISMLP_debug import *

def gather_total_data_stats(data_name, KN):
	X = np.loadtxt('data/' + data_name + '.csv', delimiter=',', dtype=np.float64)			
	Y = np.loadtxt('data/' + data_name + '_label.csv', delimiter=',', dtype=np.int32)			

	n = X.shape[0]
	d = X.shape[1]
	labels = np.unique(Y)

	print('Data : %s'%data_name)
	print('\tn : %d'%n)
	print('\td : %d'%d)
	print('\tLabels: %s'%str(labels))
	for i in labels: 
		subset = len(Y[Y == i])
		percent = subset/n
		print('\t\t%d has %d samples, %.3f%% of the total data'%(i , subset, percent ))

	print('\tSample Entry with Different labels:')
	for i in range(X.shape[0]):
		for j in range(X.shape[0]):
			Δ = np.linalg.norm(np.array(X[i,:]) - np.array(X[j,:]))
			if Δ == 0:
				if Y[i] != Y[j]:
					print('\t\t', X[i,:],Y[i], Y[j])



