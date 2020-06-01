

import sklearn.metrics
import numpy as np

def debug_rff(db, ΦX, X, σ):
	K = ΦX.dot(ΦX.T)

	γ = 1.0/(2*σ*σ)
	sk_rbf = sklearn.metrics.pairwise.rbf_kernel(X, gamma=γ)
	K2 = np.absolute(K - sk_rbf)

	print('RFF kernel')
	print(K[0:10,0:10])
	print('\n\n')
	print('RBF kernel result')
	print(sk_rbf[0:10,0:10])
	print('\n\n')
	print('Error Difference')
	print(K2[0:13,0:13])

	import pdb; pdb.set_trace()

def debug_get_Wᵦ(Xout):
	pairDistances = sklearn.metrics.pairwise.pairwise_distances(Xout)
	max_distance = np.max(pairDistances)
	σ = np.median(pairDistances)

	print('new max distance : %.3f'%max_distance)
	print('new σ : %.3f'%σ)
	γ = 1.0/(2*σ*σ)
	K = sklearn.metrics.pairwise.rbf_kernel(Xout, gamma=γ)
	
	print(K[0:10,0:10])
	print('\n\n')
	print(K[0:10,100:110])
	import pdb; pdb.set_trace()
