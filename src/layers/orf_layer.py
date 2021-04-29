

from ..libs.ISM_supervised_DR.sdr import *
from ..libs.RFF import *
from ..tools.eig_solver import PCA_by_percentage
from ..tools.distances import minmax_inter_intra_cluster_pairwise_distances
from .gaussian_layer_debug import debug_rff, debug_get_Wᵦ
import pyrfm.random_feature

#	A layer is defined by Wₐ, ℱ = ORF mapping, and maybe Wᵦ
#	ℓᴵᴺ:layer input, ℓᴼᵁᵀ : layer output from training only
#	A layer is a function f = WᵦᄋHᄋℱᄋWₐ
class orf_layer():
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
		self.get_ORF_mapping()

	def get_ORF_mapping(self):
		db = self.db
		self.XWₐ = self.ℓᴵᴺ				# same as the input since no W is used


		γ = 1/(2*self.σ*self.σ)
		self.ℱ = pyrfm.random_feature.StructuredOrthogonalRandomFeature(n_components=self.db['RFF_#_samples'], gamma=γ)		
		self.Φᵪ = self.ℱ.fit_transform(self.XWₐ)
		self.ℓᴼᵁᵀ = self.Φᵪ


	def apply_layer(self, X):
		self.Φᵪ = self.ℱ.fit_transform(X)
		return Φᵪ

