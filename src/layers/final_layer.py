
from ..tools.eig_solver import *


class final_layer():
	def __init__(self, db, last_ℓ):
		self.db = db
		self.Φᵪ = last_ℓ.ℓᴼᵁᵀ

	def apply_layer(self, X):
		return X.dot(self.Wᴼᵁᵀ)


class kpca_layer(final_layer):
	def __init__(self, db, last_ℓ):
		super(kpca_layer, self).__init__(db, last_ℓ)
		Φᵪ = self.Φᵪ
		Φᵪᵀ = Φᵪ.T

		#Φᵪ[0:10,0:20]
		#import pdb; pdb.set_trace()

		Cₒᵥ = Φᵪᵀ.dot(Φᵪ)		# biased Kernel Covariance Matrix
		self.Wᴼᵁᵀ = PCA_by_num_of_classes(Cₒᵥ, db['ɕ'])



class skpca_layer(final_layer):
	def __init__(self, db, last_ℓ):
		super(skpca_layer, self).__init__(db, last_ℓ)
		Γ = self.db['Γ']
		Φᵪ = self.Φᵪ
		Φᵪᵀ = Φᵪ.T

		Cₒᵥ = Φᵪᵀ.dot(Γ).dot(Φᵪ)		# biased Kernel Covariance Matrix
		self.Wᴼᵁᵀ = self.W = PCA_by_num_of_classes(Cₒᵥ, db['ɕ'])




#class supervised_kpca_layer(final_layer):
#	def __init__(self, db, last_ℓ):
#		super(kpca_layer, self).__init__(db, last_ℓ)
#
#		Cₒᵥ = Φᵪᵀ.dot(Φᵪ)		# biased Kernel Covariance Matrix
#		self.Wᴼᵁᵀ = PCA_by_num_of_classes(M, ɕ)
#
#	def apply_layer(self, X):
#		return X.dot(self.Wᴼᵁᵀ)
#
#
#
#
#		db = self.db
#		Φᵪ = last_ℓ.ℓᴼᵁᵀ
#		Φᵪᵀ = Φᵪ.T
#		Γ = db['Γ']
#		#n = last_ℓ.ℓᴼᵁᵀ.shape[0]
#		#H = np.eye(n) - (1.0/n)*np.ones((n,n))
#
#		Cₒᵥ = Φᵪᵀ.dot(Φᵪ)		# biased Kernel Covariance Matrix
#		biased_Cₒᵥ = Φᵪᵀ.dot(Γ).dot(Φᵪ)		# biased Kernel Covariance Matrix
#
#		#self.Wᴼᵁᵀ = Supervised_KPCA_by_percentage(Cₒᵥ, db['PCA_σᒾ%_to_keep'])
#		self.Wᴼᵁᵀ = Supervised_KPCA_by_num_of_classes(biased_Cₒᵥ, db['ɕ'])
#		#self.Wᴼᵁᵀ = Supervised_KPCA_by_num_of_classes(Cₒᵥ, db['ɕ'])
#
#		self.ℓᴼᵁᵀ = Φᵪ.dot(self.Wᴼᵁᵀ)
#		[self.Ł_目, self.ⵙ_田] = cluster_center(self.ℓᴼᵁᵀ,db['Y'])
#
#		Ł = self.get_label(self.ℓᴼᵁᵀ)
#
