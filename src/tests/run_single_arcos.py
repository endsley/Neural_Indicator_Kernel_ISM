

import numpy as np
import time
import pickle
import torch
import torch.nn.functional as F
from sklearn import preprocessing
from src.tools.DManager import *

from src.arcos.arcCosine import *
from src.networks.GP_debug import *
import sklearn.metrics



def run_single_arcos(data_name, KN, loss_function):
	db = {}
	db['data_name'] = data_name
	db['dataType'] = torch.FloatTensor				
	db["train_data_file_name"] = 'data/' + data_name + '.csv'
	db["train_label_file_name"] = 'data/' + data_name + '_label.csv'
	db["test_data_file_name"] = 'data/' + data_name + '_test.csv'
	db["test_label_file_name"] = 'data/' + data_name + '_label_test.csv'
	db['loss_function'] = loss_function
	db['batch_size'] = 10

	db['train_data'] = DManager(db["train_data_file_name"], db["train_label_file_name"], db['dataType'])
	db['train_loader'] = DataLoader(dataset=db['train_data'], batch_size=db['batch_size'], shuffle=True)
	db['test_data'] = DManager(db["test_data_file_name"], db["test_label_file_name"], db['dataType'])
	db['test_loader'] = DataLoader(dataset=db['test_data'], batch_size=db['batch_size'], shuffle=True)

	#	Construct the Network 
	network_info = {}
	network_info['d'] = db['train_data'].X.shape[1]
	network_info['c'] = len(np.unique(db['train_data'].Y))			# num of classes
	network_info['width'] = 3
	network_info['activation_function'] = "relu"	
	network_info['network_structure'] = 'None'
	network_info['knet'] = KN(db['train_data'].X, db['train_data'].Y, data_name)	
	network_info['loss_function'] = loss_function
	db['network_info'] = network_info


	#	Run network
	start_time = time.time() 			# only count the runtime code and not debug code
	for i in [0]:						# loop for indentation readability purpose only
		AC = arcCosine()
		[db['train_acc'], db['test_acc']] = AC.train_on_aKernel(db['train_data'].X, db['train_data'].Y, db['test_data'].X ,db['test_data'].Y)
		print(db['train_acc'], db['test_acc'])
	db['ΔTime'] = time.time() - start_time

	record_results_to_txt(db)


