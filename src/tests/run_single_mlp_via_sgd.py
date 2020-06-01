

import numpy as np
import time
import pickle
import torch
import torch.nn.functional as F
from sklearn import preprocessing
from src.tools.DManager import *
from src.tools.basic_optimizer import *
from src.networks.MLP import *
from src.networks.MLP_debug import *

def run_single_mlp_via_sgd(data_name, KN, loss_function):
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

	#	Use the same network depth and width as the ISMLP
	out_path = './results/' + data_name + '/mlp.pk'
	mlp = pickle.load( open( out_path, "rb" ) )
	network_layer_info = []
	for layer in mlp.layers:
		if len(layer.W.shape) > 1:
			network_layer_info.append(layer.W.shape)

	#	Construct the Network 
	network_info = {}
	network_info['d'] = db['train_data'].X.shape[1]
	network_info['c'] = len(np.unique(db['train_data'].Y))			# num of classes
	network_info['width'] = network_layer_info[1][0]			
	network_info['activation_function'] = "relu"	
	network_info['network_structure'] = network_layer_info
	network_info['knet'] = KN(db['train_data'].X, db['train_data'].Y, data_name)	
	network_info['loss_function'] = loss_function
	db['network_info'] = network_info


	#	Run network
	start_time = time.time() 			# only count the runtime code and not debug code
	for i in [0]:						# loop for indentation readability purpose only
		db['mlp'] = MLP(network_info)
		basic_optimizer(db['mlp'], db, data_loader_name='train_loader', zero_is_min=True, loss_callback=loss_function)		
	db['ΔTime'] = time.time() - start_time

	record_results_to_txt(db)


