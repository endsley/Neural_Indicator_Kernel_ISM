
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from src.networks.ISMLP_debug import *

def run_single_svm(data_name, KN):
	X = np.loadtxt('data/' + data_name + '.csv', delimiter=',', dtype=np.float64)			
	Y = np.loadtxt('data/' + data_name + '_label.csv', delimiter=',', dtype=np.int32)			
	X_test = np.loadtxt('data/' + data_name + '_test.csv', delimiter=',', dtype=np.float64)			
	Y_test = np.loadtxt('data/' + data_name + '_label_test.csv', delimiter=',', dtype=np.int32)			


	X = preprocessing.scale(X)
	X_test = preprocessing.scale(X_test)

	[out_allocation, acc, svm_object] = use_svm(X,Y)
	print(out_allocation)
	print(acc)
	import pdb; pdb.set_trace()
