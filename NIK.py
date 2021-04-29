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
	from src.tests.run_large_sample_batch import *
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
	from .src.tests.run_large_sample_batch import *
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
		self.db['network'] = ISMLP(self.db)
		[X,Y] = self.db['network'].set_input_data(self.db, X, Y)

		self.db['Yₒ'] = Allocation_2_Y(Y)		# one hot encoding version	
		self.db['ɕ'] = self.db['Yₒ'].shape[1]	# number of classes
		self.db['N'] = N = X.shape[0]
		self.db['d'] = d = X.shape[1]

		#	turn on debug with True
		self.db['record_internal'] = True					# run only for debug reasons
		self.db['print_internal_at_each_stage'] = False		# run only for debug reasons
		self.db['clear_previous_10_fold_results'] = False	# run only for debug reasons

		#	adjustable variables
		self.db['start_from_RKHS'] = False						# Project to RKHS prior to the network, will add 1st layer if True
		self.db['PCA_σᒾ_kept'] = 0.50							# 0 to 1, percentage of variance kept by ISM
		self.db['layer'] = gaussian_layer 						# gaussian_layer, gauss_kpca_layer
		#self.db['layer_feature_map_approximator'] = orf_layer 	# rff_layer, orf_layer
		self.db['layer_feature_map_approximator'] = rff_layer 	# rff_layer, orf_layer
		self.db['optimizer'] = sdr
		self.db['final_layer'] = skpca_layer
		self.db['RFF_#_samples'] = 400							# 1024	7000, 500
		self.db['exit_score'] = 0.95
		self.db['max_ℓ#'] = 30
		#self.print_initial_state()

		#	initialize folders
		path_split = data_name.split('/')
		ensure_path_exists('./results')
		if len(path_split) > 1: ensure_path_exists('./results/' + path_split[0])
		ensure_path_exists('./results/' + self.db['data_name'])
		print('Initializing Kernel Net')

	def print_initial_state(self):
		if self.db['print_internal_at_each_stage']:
			print('\tRFF_#_samples : %d'%self.db['RFF_#_samples'])
			

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
	#	pick only 1 data set	 	-------------------------
	data_name = 'wine'
	#data_name = 'random'
	#data_name = 'adversarial'
	#data_name = 'spiral'
	#data_name = 'car'
	#data_name = 'cancer'
	#data_name = 'face'
	#data_name = 'divorce'
	#data_name = 'cifar10'


	gen_10_fold_data(data_name, data_path='./data/')	


	#	pick one to run		-------------------------
	run_10_fold(data_name, kernelNet)							#  <-------- This one runs W* 
	#run_10_fold_via_sgd(data_name, kernelNet, 'ikpca')
	#run_10_fold_via_sgd(data_name, kernelNet, 'arcos')	
	#run_10_fold_via_sgd(data_name, kernelNet, 'NTK')	
	#run_10_fold_via_sgd(data_name, kernelNet, 'DeepGP')	
	#run_10_fold_via_sgd(data_name, kernelNet, 'CE_loss')
	#run_10_fold_via_sgd(data_name, kernelNet, 'MSE_loss')


	#	This will save the aggregated result to txt
	#aggregate_all_ism_results(data_name)
	#aggregate_all_sgd_results(data_name, 'CE_loss')
	#aggregate_all_sgd_results(data_name, 'MSE_loss')


