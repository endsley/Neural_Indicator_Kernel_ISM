

from ..libs.ISM_supervised_DR.sdr import *
from ..libs.RFF import *
from ..tools.eig_solver import PCA_by_percentage
from ..tools.distances import minmax_inter_intra_cluster_pairwise_distances
from .gaussian_layer_debug import debug_rff, debug_get_Wᵦ

#	A layer is defined by Wₐ, ℱ = RFF mapping, and maybe Wᵦ
#	ℓᴵᴺ:layer input, ℓᴼᵁᵀ : layer output from training only
#	A layer is a function f = WᵦᄋHᄋℱᄋWₐ
class rff_layer():
	def __init__(self, db, ℓᴵᴺ, σ):
		self.db = db
		self.σ = σ
		self.ℓᴵᴺ = ℓᴵᴺ
		self.Wₐ = self.W = np.array([])
		self.Wᵦ = np.array([])

		#	Variable that can be tweaked
		#self.db['RFF_#_samples'] = 100
		self.PCA_σᒾ_kept = 0.9

		#	Code
		self.get_RFF_mapping()

	def get_RFF_mapping(self):
		db = self.db
		self.XWₐ = self.ℓᴵᴺ				# same as the input since no W is used

		self.ℱ = RFF(sample_num=self.db['RFF_#_samples'])
		self.ℱ.initialize_RFF(db['X'], self.σ)
		
		self.Φᵪ = self.ℱ.np_feature_map(db['X'])
		self.ℓᴼᵁᵀ = self.Φᵪ


	def apply_layer(self, X):
		Φᵪ = self.ℱ.np_feature_map(X)
		return Φᵪ

