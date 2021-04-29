
import numpy as np
import os
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from src.networks.ISMLP_debug import *
from src.tools.split_10_fold import *
from src.tools.collect_HSIC_CE_MSE_results import collect_HSIC_CE_MSE_results
from src.tests.combine_kernel_img import combine_kernel_img
from src.tools.label_stats import *
from itertools import cycle

def run_large_sample_batch(data_name, KN):
	print('Running ISMLP with each layer only using a batch')

	pth_name = './data/' + data_name + '/'
	files = os.listdir(pth_name)
	batch_files = []
	for i, f in enumerate(files):
		fi = (pth_name + data_name + '_' + str(i+1) + '.csv')
		if os.path.exists(fi):
			batch_files.append(fi)

	batch_data = {}
	training_data_list = []
	batch_files.pop()	# remove the last batch for Test set

	nf = len(batch_files)		# training datasets
	for ᘐ in range(1,nf+1):
		file_name = pth_name + data_name + '_' + str(ᘐ)
		X = np.loadtxt(file_name + '.csv', delimiter=',', dtype=np.float64)			
		Y = np.loadtxt(file_name + '_label.csv', delimiter=',', dtype=np.int32)			
		X = preprocessing.scale(X)

		batch_data[ᘐ] = {}
		batch_data[ᘐ]['X'] = X[0:3000,:]
		batch_data[ᘐ]['Y'] = Y[0:3000]
		training_data_list.append(ᘐ)
		print('\tLoading %s'%file_name)

	batch_data['training_data_cyclic_list'] = cycle(training_data_list)
	batch_data['training_data_cyclic_list_itr'] = iter(batch_data['training_data_cyclic_list'])

	# test datasets
	file_name = pth_name + data_name + '_' + str(nf+1)
	X_test = np.loadtxt(file_name + '.csv', delimiter=',', dtype=np.float64)			
	X_test = preprocessing.scale(X_test)
	Y_test = np.loadtxt(file_name + '_label.csv', delimiter=',', dtype=np.int32)			
	print('\tLoading %s as the test dataset'%file_name)

	batch_data['test'] = {}
	batch_data['test']['X'] = X_test
	batch_data['test']['Y'] = Y_test

	use_path = data_name + '/' + data_name + '_1'
	knet = KN(batch_data, None, use_path)
	knet.train()

	Łabel = knet.fit(X)
	train_acc = accuracy_score(Łabel, Y)

	Ł_test = knet.fit(X_test)
	test_acc = accuracy_score(Ł_test, Y_test)


	#record_results_to_txt(train_acc, test_acc, knet)
	#combine_kernel_img(file_name)
	#record_label_stats(data_name, Y, Łabel, Ł_test)

