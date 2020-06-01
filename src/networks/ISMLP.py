#!/usr/bin/env python3

import sklearn.metrics
from .list_of_network_layers import *
from ..tools.hsic import ℍ
from ..tools.distances import cluster_center
from ..libs.ISM_supervised_DR.src.tools.classifiers import *
from .ISMLP_debug import *
from ..layers.rff_layer import *

class ISMLP(list_of_network_layers):
	def __init__(self, db):
		super(ISMLP, self).__init__(db)
		db['time_目'] = []
		self.db['ȿ_last'] = 0
		self.db['H_last'] = 0

	def initialize(self):
		db = self.db
		out_path = './results/' + db['data_name'] + '/'
		remove_files(out_path)
	

	def add_1st_ℓ(self):
		db = self.db
		if not db['start_from_RKHS']: return

		db['ℓ_begin_time'] = time.time() 			# only count the runtime code and not debug code
	
		σ = self.get_optimal_gaussian_σ(db['X'])
		new_ℓ = rff_layer(db, db['X'], σ)
		self.add_layer_to_network(new_ℓ)
		self.outer_converge(new_ℓ)

	def add_layer(self): 	# ℓ = layer
		db = self.db

		Xₒ = self.get_most_recent_training_output()	# use it as input to next layer
		σ = self.get_optimal_gaussian_σ(Xₒ)
		new_ℓ = db['layer'](db, Xₒ, σ)
		self.add_layer_to_network(new_ℓ)

		return new_ℓ

	def outer_converge(self, new_ℓ):
		db = self.db

		#db['ȿ'] = sklearn.metrics.silhouette_score(new_ℓ.Φᵪ, db['Y'])	# in RKHS
		db['ȿ'] = sklearn.metrics.silhouette_score(new_ℓ.XWₐ, db['Y'])	# in IDS 
		db['ΔTime'] = time.time() - db['ℓ_begin_time']


		db['hsic'] = ℍ(new_ℓ.Φᵪ, db['Yₒ'])
		print_each_ℓ_ℹ(self, new_ℓ)									#	debug only
		record_each_ℓ(self, new_ℓ)									#	debug only


		if db['exit_criteria'] == 'HSIC':
			if hsic >= self.db['exit_score']: return True
			self.db['H_last'] = hsic
		else:
			if np.absolute(self.db['ȿ'] - self.db['ȿ_last']) < 0.001 and self.db['ȿ'] > 0.80: return True
			if self.db['ȿ'] >= self.db['exit_score']: return True
			self.db['ȿ_last'] = self.db['ȿ']


		return False

	def get_label(self, X):
		Ꮬð田 = sklearn.metrics.pairwise.pairwise_distances(X, self.ⵙ_田, metric='euclidean')
		closest_Ł = np.argmin(Ꮬð田,axis=1)
		Ł = self.Ł_目[closest_Ł]	# getting the label
		return Ł

	def add_final_layer(self, last_ℓ):	
		self.db['ℓ_begin_time'] = time.time() 								#	clock begin

		self.final_ℓ = self.db['final_layer'](self.db, last_ℓ)
		ℓᴼᵁᵀ = self.final_ℓ.apply_layer(last_ℓ.ℓᴼᵁᵀ)
		[self.Ł_目, self.ⵙ_田] = cluster_center(ℓᴼᵁᵀ, self.db['Y'])

		self.db['ΔTime'] = time.time() - self.db['ℓ_begin_time']			# clock end

		print_final_ℓ(self, last_ℓ.ℓᴼᵁᵀ)									#	debug only
		record_final_ℓ(self, last_ℓ.ℓᴼᵁᵀ)									#	debug only



	def fit(self, X):
		ℓᴵᴺ = X
		for ㄢ, ℓ in enumerate(self.ℓ_目):
			ℓᴼᵁᵀ = ℓ.apply_layer(ℓᴵᴺ)
			ℓᴵᴺ = ℓᴼᵁᵀ
	
		Netᴼᵁᵀ = self.final_ℓ.apply_layer(ℓᴼᵁᵀ)
		return self.get_label(Netᴼᵁᵀ)
		#return self.get_label(ℓᴼᵁᵀ)


