#!/usr/bin/env python


import warnings
warnings.filterwarnings("ignore")

import sys
import matplotlib
import time
import numpy as np
import matplotlib.pyplot as plt
import random
import itertools
import socket
from sklearn import preprocessing
import sklearn.metrics

from src.tools.eig_solver import *
from src.libs.RFF import *
from src.tools.distances import mean_intra_cluster_pairwise_distances
import NSM_debug

# For a complete list : https://docs.google.com/spreadsheets/d/1XajDsrXjXy_0RFDW5ru-4-YDJFsSCVcCjtZqhvm8Eq4/edit?usp=sharing
# ⵙ 	: within cluster
# Ꮬ 	: between cluster
# 取	: get
# ð		: dimension
# Ո		: intersection
# Ճ田	: projection matrix
# Ɲ		: column vectors for the null space of within clusters Aᵢﺯ
# ʆ		: column vectors for the span of between clusters Aᵢﺯ
# ㆀ	: pair
# ⵌ		: number

np.set_printoptions(precision=4)
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=300)
np.set_printoptions(suppress=True)


class NSM():
	def __init__(self, X, Y, map2RKHS=True):	#	X=data, Y=label, q=reduced dimension
		#	automated variables
		self.db = {}
		self.db['X'] = X
		self.db['Y'] = Y
		self.db['σₐ'] = mean_intra_cluster_pairwise_distances(X, Y)
		self.Ł_目 = np.unique(Y)
		self.db['c'] = len(self.Ł_目)
		self.RFF_ð = 200


		ℱᴀ = RFF(sample_num=self.RFF_ð)
		ℱᴀ.initialize_RFF(X, self.db['σₐ'])
		if map2RKHS: self.db['X'] = ℱᴀ.np_feature_map(X)
		self.ℱᴀ = ℱᴀ
		self.ℱʙ = None

	def __del__(self):
		self.db.clear()
	
	def train(self):		 
		db = self.db
		X = db['X']
		Y = db['Y']
		ɲ = X.shape[0]
		ð = X.shape[1]
		Ł_目 = self.Ł_目

		Aᵢﺯ_ⵙ = np.zeros((ð,ð))
		
		self.ⵌ_ᵢﺯ_ㆀ_ⵙ = 0 
		for i in Ł_目:
			indices = np.where(Y == i)
			subX = X[indices, :][0]
			Aᵢﺯ_ⵙ += self.取_Aᵢﺯ(subX)
			self.ⵌ_ᵢﺯ_ㆀ_ⵙ += subX.shape[0]*subX.shape[0]

		self.ⵌ_ᵢﺯ_ㆀ_Ꮬ = ɲ*ɲ - self.ⵌ_ᵢﺯ_ㆀ_ⵙ
		Aᵢﺯ_Ꮬ = self.取_Aᵢﺯ(X) - Aᵢﺯ_ⵙ
		self.ƝՈʆ = self.取Ո(Aᵢﺯ_ⵙ, Aᵢﺯ_Ꮬ)


	def get_reduced_dim_data(self, X):
		db = self.db

		ϰ = self.ℱᴀ.np_feature_map(X).dot(self.ƝՈʆ)
		if self.ℱʙ is None:
			self.ℱʙ = RFF(sample_num=self.RFF_ð)
			self.ℱʙ.initialize_RFF(ϰ, db['σᵦ'])

		Φϰ = self.ℱʙ.np_feature_map(ϰ)
		Wᵦ = PCA_by_num_of_classes(Φϰ.T.dot(Φϰ), db['c'])

		Xᴏᶙⲧ = Φϰ.dot(Wᵦ)
		return Xᴏᶙⲧ

	def 取Ո(self, Aᵢﺯ_ⵙ, Aᵢﺯ_Ꮬ):
		db = self.db
		Ɲ = get_Null_space(Aᵢﺯ_ⵙ)
		ʆ = get_Principal_Components(Aᵢﺯ_Ꮬ)

		Ճ = Ɲ.T.dot(ʆ)	# project the span onto the null space
		ƝՃ = Ɲ.dot(Ճ)	# discover intersection
		self.ƝՃ = ƝՃ
		db['σᵦ'] = self.get_optimal_σ(ƝՃ)

		return ƝՃ


	def get_optimal_σ(self, ƝՃ):
		db = self.db
		X = db['X'].dot(ƝՃ)
		Y = db['Y']
		ɲ = X.shape[0]
		ð = X.shape[1]
		Ł_目 = self.Ł_目
		self.ϰ目 = np.empty((0, ð))
		ㆀ_ɗᴍₐᵪ = 0	

		for i in Ł_目:
			indices = np.where(Y == i)
			ϰ = X[indices, :][0]
			ϰˊ = np.mean(ϰ, axis=0)
			self.ϰ目 = np.vstack((self.ϰ目, ϰˊ))
			
			目ɗᴍₐᵪ = np.max(sklearn.metrics.pairwise.pairwise_distances(ϰ))
			if 目ɗᴍₐᵪ > ㆀ_ɗᴍₐᵪ: ㆀ_ɗᴍₐᵪ = 目ɗᴍₐᵪ
		
		return 2*ㆀ_ɗᴍₐᵪ


	def 取_Aᵢﺯ_not_vectorized(X):
		n = X.shape[0]
		ð = X.shape[1]
	
		start = time.time()
		Aᵢﺯ = np.zeros((ð, ð))
		for i in np.arange(n):
			for j in np.arange(n):
				xᵢ = np.reshape(X[i,:], (ð, 1))
				xﺯ = np.reshape(X[j,:], (ð, 1))
				Δx = xᵢ - xﺯ
	
				Aᵢﺯ += Δx.dot(Δx.T)

		return Aᵢﺯ


	def 取_Aᵢﺯ(self, X):
		n = X.shape[0]
		ð = X.shape[1]
		Aᵢﺯ = np.zeros((ð, ð))

		ᕼ = np.eye(n) - (1.0/n)*np.ones((n,n))
		HX = ᕼ.dot(X)
		Aᵢﺯ = 2*n*HX.T.dot(HX)

		return Aᵢﺯ





if __name__ == "__main__":
	data_name = 'wine_1'
	X = np.loadtxt('data/' + data_name + '.csv', delimiter=',', dtype=np.float64)			
	Y = np.loadtxt('data/' + data_name + '_label.csv', delimiter=',', dtype=np.int32)			
	X_test = np.loadtxt('data/' + data_name + '_test.csv', delimiter=',', dtype=np.float64)			
	Y_test = np.loadtxt('data/' + data_name + '_label_test.csv', delimiter=',', dtype=np.int32)			


	X = preprocessing.scale(X)
	X_test = preprocessing.scale(X_test)


	nsm = NSM(X,Y)
	nsm.train()

	Xˊ = nsm.get_reduced_dim_data(X)
	[out_allocation, training_acc, svm_object] = use_svm(Xsmall, Y, k='rbf')






#	s = NSM(X,Y,q=7)	#q if not set, it is automatically set to 80% of data variance by PCA
#
#
#	s.train()
#	W = s.get_projection_matrix()
#	Xsmall = s.get_reduced_dim_data(X)
#
#	[out_allocation, training_acc, svm_object] = use_svm(Xsmall, Y, k='rbf')
#	test_acc = apply_svm(X_test.dot(W), Y_test, svm_object)
#
#	print('Using : %s '%type(s.db['kernel']).__name__)
#	print('\tDataset : %s'%(data_name))
#	print('\tInput dimension : %d x %d'%(X.shape[0],X.shape[1]))
#	print('\tOutput dimension : %d x %d'%(Xsmall.shape[0],Xsmall.shape[1]))
#	print('\tInitial HSIC : %.4f'%s.db['init_HSIC'])
#	print('\tFinal HSIC : %.4f'%s.db['final_HSIC'])
#	print('\tTraining Accuracy : %.4f'%training_acc)
#	print('\tTest Accuracy : %.4f'%test_acc)
#
#
#	del s

