#!/usr/bin/env python3

import numpy as np
import sklearn.metrics
from ..tools.distances import mean_intra_cluster_pairwise_distances
from ..libs.opt_gaussian import *
import pickle
import time 


class list_of_network_layers():
	def __init__(self, db):
		self.db = db
		self.ℓ_目 = []	# layer list
		self.Ɗ = {}

	def get_optimal_gaussian_σ(self, X):
		σ = get_opt_σ(X, self.db['Y'])
		#σ = mean_intra_cluster_pairwise_distances(X, self.db['Y'])	# deprecated version
		#σ = 0.3
		return σ

		
	def get_most_recent_training_output(self):
		if len(self.ℓ_目) == 0:
			return self.db['X']
		else:
			return self.ℓ_目[-1].ℓᴼᵁᵀ

	def add_layer_to_network(self, new_layer):
		self.ℓ_目.append(new_layer)

		

	def save_network(self, out_path):
		tmp_db = self.db
		self.db = {}
		self.layers = self.ℓ_目

		for ℓ in self.layers:	# delete stuff within each layer
			ℓ.ℓᴵᴺ = None
			ℓ.XWₐ = None
			ℓ.Φᵪ = None
			ℓ.ℓᴼᵁᵀ = None
			ℓ.db = None

		self.final_ℓ.db = None
		self.final_ℓ.Φᵪ = None

		pickle.dump( self, open( out_path, "wb" ) )
		self.db = tmp_db
