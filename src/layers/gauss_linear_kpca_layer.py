

from ..libs.ISM_supervised_DR.sdr import *
from ..libs.RFF import *
from ..tools.eig_solver import PCA_by_percentage
from ..tools.hsic import ℍ
from .gaussian_layer_debug import debug_rff, debug_get_Wᵦ

#	A layer is defined by Wₐ, ℱ = RFF mapping, and maybe Wᵦ
#	ℓᴵᴺ:layer input
#	A layer is a function f = HᄋℱᄋWₐ
class gauss_linear_kpca_layer():
	def __init__(self, db, ℓᴵᴺ, σ):
		self.db = db
		self.σ = σ
		self.ℓᴵᴺ = ℓᴵᴺ
		self.pc = db['PCA_σ2%_to_keep']
		self.db['RFF_#_samples'] = 500

		self.get_Wₐ_via_ISM()
		self.get_RFF_mapping()
		self.get_Wᵦ()


	def get_Wₐ_via_ISM(self):
		db = self.db
		network_model = sdr(self.ℓᴵᴺ, db['Y'], var_percentage=self.pc, σ=self.σ)
		network_model.train()

		if 'Γ' not in db: db['Γ'] = network_model.db['Γ']
		if 'H' not in db: db['H'] = network_model.db['H']

		self.Y = network_model.db['Y']
		self.Wₐ = network_model.get_projection_matrix()


	def get_RFF_mapping(self):
		db = self.db
		self.XWₐ = self.ℓᴵᴺ.dot(self.Wₐ)

		ℱ = RFF(sample_num=self.db['RFF_#_samples'])
		ℱ.initialize_RFF(self.XWₐ, self.σ)
		self.ℱ = ℱ
		
		self.Φᵪ = self.ℱ.np_feature_map(self.XWₐ)
		self.HΦᵪ = db['H'].dot(self.Φᵪ)

		self.ᕼℊ = ℍ(self.HΦᵪ, self.Y)		# Gaussian Layer
		self.ᕼℓ = ℍ(self.ℓᴵᴺ, self.Y)		# Linear Layer

		#self.ᕼℊ = np.sqrt(self.ᕼℊ/(self.ᕼℊ + self.ᕼℓ))
		#self.ᕼℓ = np.sqrt(self.ᕼℓ/(self.ᕼℊ + self.ᕼℓ))
		self.ɣₐ = self.ᕼℊ/(self.ᕼℊ + self.ᕼℓ)
		self.ɣᵦ = self.ᕼℓ/(self.ᕼℊ + self.ᕼℓ)

		print(self.ɣₐ)
		print(self.ɣᵦ)

		self.HΦᵪⵜℓᴵᴺ = np.hstack((np.sqrt(self.ɣₐ)*self.HΦᵪ, np.sqrt(self.ɣᵦ)*self.ℓᴵᴺ))

		#self.ℓᴼᵁᵀ = np.hstack((self.ᕼℊ*self.HΦᵪ, self.ᕼℓ*self.ℓᴵᴺ))

	def get_Wᵦ(self):
		Φᐤ = self.HΦᵪⵜℓᴵᴺ
		Φᐤᵀ = Φᐤ.T
		Γ = self.db['Γ']

		Cₒᵥ = Φᐤᵀ.dot(Φᐤ)		# biased Kernel Covariance Matrix
		biased_Cₒᵥ = Φᐤᵀ.dot(Γ).dot(Φᐤ)		# biased Kernel Covariance Matrix
		self.Wᵦ = PCA_by_percentage(Cₒᵥ, biased_Cₒᵥ, self.pc)
		self.ℓᴼᵁᵀ = Φᐤ.dot(self.Wᵦ)

	def apply_layer(self, X):
		XWₐ = X.dot(self.Wₐ)
		Φᵪ = self.ℱ.np_feature_map(XWₐ)
		HΦᵪ = self.H.dot(Φᵪ)


		ᕼℊ = ℍ(HΦᵪ, Y)		# Gaussian Layer
		ᕼℓ = ℍ(self.ℓᴵᴺ, Y)		# Linear Layer
		HΦᵪⵜℓᴵᴺ = np.hstack((ᕼℊ*HΦᵪ, ᕼℓ*X))

		ℓᴼᵁᵀ = HΦᵪⵜℓᴵᴺ.dot(self.Wᵦ)
		return ℓᴼᵁᵀ

