

from ..libs.ISM_supervised_DR.sdr import *
from ..libs.RFF import *
from ..tools.eig_solver import PCA_by_percentage
from ..tools.distances import minmax_inter_intra_cluster_pairwise_distances
from .gaussian_layer_debug import debug_rff, debug_get_Wᵦ

#	A layer is defined by Wₐ, ℱ = RFF mapping, and maybe Wᵦ
#	ℓᴵᴺ:layer input, ℓᴼᵁᵀ : layer output from training only
#	A layer is a function f = WᵦᄋHᄋℱᄋWₐ
class gaussian_layer():
	def __init__(self, db, ℓᴵᴺ, σ):
		self.db = db
		self.σ = σ
		self.ℓᴵᴺ = ℓᴵᴺ
		self.Wᵦ = np.array([])

		#	Variable that can be tweaked
		#self.db['RFF_#_samples'] = 100


		#	Code
		self.get_Wₐ_via_ISM()
		self.get_RFF_mapping()


	def get_Wₐ_via_ISM(self):
		db = self.db
		optimizer = db['optimizer'](self.ℓᴵᴺ, db['Y'], var_percentage=db['PCA_σᒾ_kept'], σ=self.σ)
		optimizer.train()

		if 'Γ' not in db: db['Γ'] = optimizer.db['Γ']
		if 'H' not in db: db['H'] = optimizer.db['H']

		self.Wₐ = self.W = optimizer.get_projection_matrix()

	def get_RFF_mapping(self):
		db = self.db
		self.XWₐ = self.ℓᴵᴺ.dot(self.Wₐ)

		ℱ = RFF(sample_num=self.db['RFF_#_samples'])
		ℱ.initialize_RFF(self.XWₐ, self.σ)
		self.ℱ = ℱ
		
		self.Φᵪ = self.ℱ.np_feature_map(self.XWₐ)
		self.ℓᴼᵁᵀ = self.Φᵪ


	def apply_layer(self, X):
		XWₐ = X.dot(self.Wₐ)
		Φᵪ = self.ℱ.np_feature_map(XWₐ)
		return Φᵪ

