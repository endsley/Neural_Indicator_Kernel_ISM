
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from src.networks.ISMLP_debug import *
from src.tests.combine_kernel_img import combine_kernel_img

def run_single(data_name, KN):
	X = np.loadtxt('data/' + data_name + '.csv', delimiter=',', dtype=np.float64)			
	Y = np.loadtxt('data/' + data_name + '_label.csv', delimiter=',', dtype=np.int32)			
	X_test = np.loadtxt('data/' + data_name + '_test.csv', delimiter=',', dtype=np.float64)			
	Y_test = np.loadtxt('data/' + data_name + '_label_test.csv', delimiter=',', dtype=np.int32)			

	X = preprocessing.scale(X)
	X_test = preprocessing.scale(X_test)

	knet = KN(X,Y, data_name)	#q if not set, it is automatically set to 80% of data variance by PCA
	knet.train()

	Łabel = knet.fit(X)
	train_acc = accuracy_score(Łabel, Y)

	Ł_test = knet.fit(X_test)
	test_acc = accuracy_score(Ł_test, Y_test)

	record_results_to_txt(train_acc, test_acc, knet)
	combine_kernel_img(data_name)
	import pdb; pdb.set_trace()
