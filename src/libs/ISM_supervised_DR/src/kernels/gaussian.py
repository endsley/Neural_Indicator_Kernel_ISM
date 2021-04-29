

import numpy as np
import sklearn.metrics
import time
from ..tools import kernel_lib as klib
import pyrfm.random_feature


class gaussian():
	def __init__(self, db):
		self.db = db
		n = db['X'].shape[0]
		self.orf_kick_in_at = 16000
		#self.orf_kick_in_at = 10

		if db['σ'] is None:
			self.σ = np.median(sklearn.metrics.pairwise.pairwise_distances(db['X']))
		else:
			self.σ = db['σ']

		if n > self.orf_kick_in_at:
			γ = 1/(2*self.σ*self.σ)
			self.sorf = pyrfm.random_feature.StructuredOrthogonalRandomFeature(n_components=2048, gamma=γ)

	def __del__(self):
		pass

	def get_kernel_matrix(self, W):
		db = self.db

		n = db['X'].shape[0]
		X = db['X']
		σ = self.σ
	
		if n < self.orf_kick_in_at:
			#t1 = time.time()
			Kx = klib.rbk_sklearn(X.dot(W), σ)
			#t2 = time.time() - t1
			#print(t2)
		else:
			#t1 = time.time()
			Φx = self.sorf.fit_transform(X.dot(W))
			Kx = Φx.dot(Φx.T)
			np.fill_diagonal(Kx, 0)
			#t2 = time.time() - t1
			#print(t2)
			
		return Kx

	def get_Φ(self, W): # using the smallest eigenvalue 
		db = self.db

		X = db['X']
		σ = self.σ
		Γ = db['Γ']
	
		#Kx = klib.rbk_sklearn(X.dot(W), σ)
		Kx =  self.get_kernel_matrix(W)

		Ψ=Γ*Kx
		D_Ψ = klib.compute_Degree_matrix(Ψ)
		Φ = X.T.dot(D_Ψ - Ψ).dot(X) 			#debug.compare_Φ(db, Φ, Ψ)	
		return Φ

	def get_Φ0(self):	# using the smallest eigenvalue 
		db = self.db 
		X = db['X']
		D_γ = klib.compute_Degree_matrix(db['Γ'])
		Φ = X.T.dot(D_γ - db['Γ']).dot(X); 			#debug.compare_Φ(db, Φ, Ψ)
		return Φ

