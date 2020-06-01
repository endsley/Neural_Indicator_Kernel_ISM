

from ..libs.ISM_supervised_DR.sdr import *
from ..libs.RFF import *
from ..tools.eig_solver import PCA_by_percentage
from .gaussian_layer_debug import debug_rff, debug_get_Wᵦ

#	A layer is defined by Wₐ, ℱ = RFF mapping, and maybe Wᵦ
#	ℓᴵᴺ:layer input
#	A layer is a function f = WᵦᄋHᄋℱᄋWₐ
class gauss_kpca_layer():
	def __init__(self, db, ℓᴵᴺ, σ):
		self.db = db
		self.σ = σ
		self.ℓᴵᴺ = ℓᴵᴺ

		#	Variable that can be tweaked
		self.Wₐ_PCA_σᒾ_kept = 0.70
		self.Wᵦ_PCA_σᒾ_kept = 0.99
		self.db['RFF_#_samples'] = 400


		#	Code
		self.get_Wₐ_via_ISM()
		self.get_RFF_mapping()
		self.get_Wᵦ()


	def get_Wₐ_via_ISM(self):
		db = self.db
		network_model = sdr(self.ℓᴵᴺ, db['Y'], var_percentage=self.Wₐ_PCA_σᒾ_kept, σ=self.σ)
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


	def get_Wᵦ(self):
		Φᵪ = self.Φᵪ
		#HΦᵪ = self.HΦᵪ
		Φᵪᵀ = self.Φᵪ.T
		Γ = self.db['Γ']

		Cₒᵥ = Φᵪᵀ.dot(Φᵪ)		# biased Kernel Covariance Matrix
		#Cₒᵥ = Φᵪᵀ.dot(Γ).dot(Φᵪ)		# biased Kernel Covariance Matrix
		self.Wᵦ = PCA_by_percentage(Cₒᵥ, self.Wᵦ_PCA_σᒾ_kept)
		self.ℓᴼᵁᵀ = Φᵪ.dot(self.Wᵦ)
		

	def apply_layer(self, X):	
		XWₐ = X.dot(self.Wₐ)
		Φᵪ = self.ℱ.np_feature_map(XWₐ)
		n = Φᵪ.shape[0]
		H = np.eye(n) - (1.0/n)*np.ones((n,n))

		#HΦᵪ = H.dot(Φᵪ)
		ℓᴼᵁᵀ = Φᵪ.dot(self.Wᵦ)

		return ℓᴼᵁᵀ

