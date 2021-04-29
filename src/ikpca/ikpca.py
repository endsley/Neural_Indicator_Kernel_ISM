#!/usr/bin/env python

import numpy as np
import math
from numpy import genfromtxt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits
from sklearn.decomposition import KernelPCA


class ikpca():
	def __init__(self):
		self.πˉᑊ = 1/np.pi


	def acKernel(self, x,y,l,n=1):
		return self.Kᵣ(x,y,l,n)

	def J(self, θ, n=1): 
		if n == 1:
			return np.sin(θ) + (np.pi - θ)*np.cos(θ)

	def Kᵣ(self, x,y,l,n=1): # assume n = 1 for now
		J = self.J
		πˉᑊ = self.πˉᑊ
		Kᵣ = self.Kᵣ

		if l == 1:
			ㅣxㅣ = np.linalg.norm(x)
			ㅣyㅣ = np.linalg.norm(y)
			ㅣxㅣㅣyㅣ = ㅣxㅣ*ㅣyㅣ
			xᵀy = x.dot(y)

			R = xᵀy/ㅣxㅣㅣyㅣ
			if R > 1: R = 1

			θ = np.arccos(R)
			Kᵢﺫ = πˉᑊ*ㅣxㅣㅣyㅣ*J(θ)
		elif l > 1:
			KₓₓKᵧᵧ = np.sqrt(Kᵣ(x,x,l-1,n)*Kᵣ(y,y,l-1,n))
			Kₓᵧ = Kᵣ(x,y,l-1)
			θ = np.arccos(Kₓᵧ/KₓₓKᵧᵧ)
			Kᵢﺫ = πˉᑊ*KₓₓKᵧᵧ*J(θ)

		if math.isnan(Kᵢﺫ):
			import pdb; pdb.set_trace()

		return Kᵢﺫ

	def train(self, X_train, Y_train, X_test, Y_test):
		T1 = KernelPCA(kernel='rbf')
		X1 = T1.fit_transform(X_train)
		
		T2 = KernelPCA(kernel='rbf')
		X2 = T2.fit_transform(X1)

		T3 = KernelPCA(n_components=10,kernel='rbf')
		X3 = T3.fit_transform(X1)

		clf = SVC(C = 1, cache_size = 100000)
		clf.fit(X3, Y_train)
		Ł_train = clf.predict(X3)


		X1 = T1.fit_transform(X_test)
		X2 = T2.fit_transform(X1)
		X3 = T3.fit_transform(X2)

		Ł_test = clf.predict(X3)

		train_acc = accuracy_score(Ł_train, Y_train)
		test_acc = accuracy_score(Ł_test, Y_test)

		return [train_acc, test_acc]


if __name__ == "__main__":
	X = genfromtxt('dat/wine_2.csv', delimiter=',')
	Y = genfromtxt('dat/wine_2_label.csv', delimiter=',')
	ẋ = genfromtxt('dat/wine_2_test.csv', delimiter=',')
	ý = genfromtxt('dat/wine_2_label_test.csv', delimiter=',')
	
	IKpca = ikpca()
	[train_acc, test_acc] = IKpca.train(X,Y, ẋ, ý)

	print([train_acc, test_acc])
