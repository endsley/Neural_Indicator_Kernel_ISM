#!/usr/bin/env python

import warnings
warnings.filterwarnings("ignore")

import sys
import matplotlib
import numpy as np
import random
import itertools
import socket


if __name__ == "__main__":
	from src.networks.ISMLP import *
	from src.networks.ISMLP_debug import *
	from src.layers.gaussian_layer import *
	from src.layers.gauss_kpca_layer import *
	from src.layers.gauss_linear_kpca_layer import *
	from src.layers.final_layer import *
	from src.libs.ISM_supervised_DR.src.tools.kernel_lib import Allocation_2_Y
	from src.libs.ISM_supervised_DR.src.tools.kernel_lib import Y_2_allocation
	from src.libs.ISM_supervised_DR.src.tools.kernel_lib import rank_by_variance
	from src.libs.ISM_supervised_DR.sdr import *
	from src.tools.distances import mean_intra_cluster_pairwise_distances
	from src.tools.split_10_fold import *
	from src.tools.collect_HSIC_CE_MSE_results import collect_HSIC_CE_MSE_results
	from src.tests.run_single import run_single
	from src.tests.run_single_mlp_via_sgd import run_single_mlp_via_sgd
	from src.tests.run_single_svm import run_single_svm
	from src.tests.run_10_fold import *
	from src.tests.run_10_fold_via_sgd import *
	from src.tests.combine_kernel_img import combine_kernel_img
	from src.tasks.gather_data_stats import *
else:
	from .src.networks.ISMLP import *
	from .src.networks.ISMLP_debug import *
	from .src.layers.gaussian_layer import *
	from .src.layers.gauss_kpca_layer import *
	from .src.layers.gauss_linear_kpca_layer import *
	from .src.layers.final_layer import *
	from .src.libs.ISM_supervised_DR.src.tools.kernel_lib import Allocation_2_Y
	from .src.libs.ISM_supervised_DR.src.tools.kernel_lib import Y_2_allocation
	from .src.libs.ISM_supervised_DR.src.tools.kernel_lib import rank_by_variance
	from .src.libs.ISM_supervised_DR.sdr import *
	from .src.tools.distances import mean_intra_cluster_pairwise_distances
	from .src.tools.split_10_fold import *
	from .src.tools.collect_HSIC_CE_MSE_results import collect_HSIC_CE_MSE_results
	from .src.tests.run_single import run_single
	from .src.tests.run_single_mlp_via_sgd import run_single_mlp_via_sgd
	from .src.tests.run_single_svm import run_single_svm
	from .src.tests.run_10_fold import *
	from .src.tests.run_10_fold_via_sgd import *
	from .src.tests.combine_kernel_img import combine_kernel_img
	from .src.tasks.gather_data_stats import *



np.set_printoptions(precision=4)
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=300)
np.set_printoptions(suppress=True)


class kernelNet():
	def __init__(self, X, Y, data_name=''):	#	X=data, Y=label, q=reduced dimension
		#	automated variables
		self.db = {}
		self.db['data_name'] = data_name
		self.db['X'] = X
		self.db['Y'] = Y				
		self.db['Yₒ'] = Allocation_2_Y(Y)		# one hot encoding version
		self.db['ɕ'] = self.db['Yₒ'].shape[1]	# number of classes
		self.db['N'] = N = X.shape[0]
		self.db['d'] = d = X.shape[1]

		#	turn on debug with True
		self.db['record_internal'] = True					# run only for debug reasons
		self.db['print_internal_at_each_stage'] = False		# run only for debug reasons


		#	adjustable variables
		self.db['start_from_RKHS'] = False					# Project to RKHS prior to the network, will add 1st layer if True
		self.db['PCA_σᒾ_kept'] = 0.90						# 0 to 1, percentage of variance kept by ISM
		self.db['network'] = ISMLP(self.db)
		self.db['layer'] = gaussian_layer 					# gaussian_layer, gauss_kpca_layer
		self.db['optimizer'] = sdr
		self.db['final_layer'] = skpca_layer
		self.db['RFF_#_samples'] = 300			
		self.db['exit_criteria'] = 'silhouette'					# options : 'HSIC', 'silhouette'
		#self.db['RFF_#_samples'] = 2100						# debug : make sure to comment out
		#self.db['exit_score'] = 0.9999
		self.db['exit_score'] = 0.95
		self.db['max_ℓ#'] = 30

		#	initialize folders
		path_split = data_name.split('/')
		ensure_path_exists('./results')
		if len(path_split) > 1: ensure_path_exists('./results/' + path_split[0])
		ensure_path_exists('./results/' + self.db['data_name'])


	def __del__(self):
		del self.db['network']
		self.db.clear()
	
	def train(self):
		db = self.db
		net = db['network']
		net.initialize()
		#net.add_1st_ℓ()

		for Ꮻ in np.arange(db['max_ℓ#']):				# maximum of 30 layers
			db['ℓ_begin_time'] = time.time() 			# only count the runtime code and not debug code
			new_ℓ = net.add_layer()
			if net.outer_converge(new_ℓ): break;

		net.add_final_layer(new_ℓ)
		
	def fit(self, X):
		return self.db['network'].fit(X)


if __name__ == "__main__":
#	for run_10_fold 
	data_name = 'wine'
	#data_name = 'cancer'
	#data_name = 'car'
	#data_name = 'face'
	#data_name = 'divorce'
	#data_name = 'spiral'
	#data_name = 'random'
	#data_name = 'adversarial'

	#gather_total_data_stats(data_name, kernelNet)
	#collect_HSIC_CE_MSE_results(data_name)
	#gen_10_fold_data(data_name, data_path='./data/')	
	#aggregate_all_ism_results(data_name)
	#aggregate_all_sgd_results(data_name, 'CE_loss')
	#aggregate_all_sgd_results(data_name, 'MSE_loss')

	run_10_fold(data_name, kernelNet)								#	< --  This is HSIC
	#run_10_fold_via_sgd(data_name, kernelNet, 'CE_loss')			#	< --  This is CE
	#run_10_fold_via_sgd(data_name, kernelNet, 'MSE_loss')			#	< --  This is MSE


#	for run_single particular dataset within 10 fold
	#data_name = 'random/random_1'
	#data_name = 'adversarial/adversarial_1'
	#data_name = 'car/car_1'
	#data_name = 'face/face_1'
	#data_name = 'cancer/cancer_3'
	#data_name = 'wine/wine_1'
	#data_name = 'divorce/divorce_2'

	#run_single_mlp_via_sgd(data_name, kernelNet, 'CE_loss')		# options : CE_loss, MSE_loss
	#run_single_mlp_via_sgd(data_name, kernelNet, 'MSE_loss')		# options : CE_loss, MSE_loss
	#run_single_svm(data_name, kernelNet)
	#run_single(data_name, kernelNet)


##	run 10 fold on all datasets (Avoid running this one unless you know what you are doing)
#	data_list = ['wine', 'cancer', 'car', 'face','divorce', 'random', 'adversarial', 'spiral']
#	#data_list = ['wine', 'car', 'face','divorce']
#	#data_list = ['scale']
#
#	for data_name in data_list:
#		collect_HSIC_CE_MSE_results(data_name)
#
#	for data_name in data_list:
#		run_10_fold(data_name, kernelNet)
#		run_10_fold_via_sgd(data_name, kernelNet, 'CE_loss')
#		run_10_fold_via_sgd(data_name, kernelNet, 'MSE_loss')
#
#	combine_kernel_img(data_list)
