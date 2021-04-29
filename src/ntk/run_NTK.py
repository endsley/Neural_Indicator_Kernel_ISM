#!/usr/bin/env python

import argparse
import os
import math
import numpy as np
#import tools
from numpy import genfromtxt

from src.ntk.NTK import *
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


def run_NTK(X,Y, ẋ, ý):	#Train: X, Y, Test: ẋ, ý
	MAX_DEP = 3
	DEP_LIST = list(range(MAX_DEP))
	
	Xbig = np.vstack((X, ẋ))
	Ybig = np.hstack((Y, ý))
	
	Ks = kernel_value_batch(Xbig, MAX_DEP)
	K = Ks[2][2]
	clf = SVC(kernel = "precomputed", C = 1, cache_size = 100000)

	K_train = K[0:X.shape[0], 0:X.shape[0]]
	K_test = K[X.shape[0]:Xbig.shape[0], 0:X.shape[0]]

	clf.fit(K_train, Y)
	Ł_train = clf.predict(K_train)
	Ł_test = clf.predict(K_test)

	train_acc = accuracy_score(Ł_train, Y)
	test_acc = accuracy_score(Ł_test, ý)

	return [train_acc, test_acc]

if __name__ == "__main__":
	X = genfromtxt('data/wine_2.csv', delimiter=',')
	Y = genfromtxt('data/wine_2_label.csv', delimiter=',')
	ẋ = genfromtxt('data/wine_2_test.csv', delimiter=',')
	ý = genfromtxt('data/wine_2_label_test.csv', delimiter=',')
	
	[train_acc, test_acc] = run_NTK(X,Y, ẋ, ý)



