#!/usr/bin/env python

import numpy as np


def use_num_of_samples_per_class(num_samples):
	file_name = 'mnist'
	
	X = np.loadtxt(file_name + '.csv', delimiter=',', dtype=np.float64)			
	Y = np.loadtxt(file_name + '_label.csv', delimiter=',', dtype=np.int32)			
	label_list = np.unique(Y)

	Ẋ = np.empty((0, X.shape[1]))
	Ý = np.empty(0)

	for l in label_list:
		Xsub = X[Y == l,:]
		Ysub = Y[Y == l]

		randIndices = np.round(Xsub.shape[0]*np.random.rand(num_samples)).astype(int)
		Ẋ = np.vstack((Ẋ, Xsub[randIndices,:]))
		Ý = np.hstack((Ý, Ysub[randIndices]))

	print(Ẋ.shape)
	print(Ý.shape)

	np.savetxt(file_name + '_small.csv', Ẋ, delimiter=',', fmt='%f') 
	np.savetxt(file_name + '_small_label.csv', Ý, delimiter=',', fmt='%d') 

use_num_of_samples_per_class(10)
