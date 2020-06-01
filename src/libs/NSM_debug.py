

import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics

def study_ƝՃ(db, ƝՃ):
	ⵙ_目 = study_ƝՃ_為ⵙ(db, ƝՃ)
	Ꮬ_目 = study_ƝՃ_為Ꮬ(db, ƝՃ)


	#σ = np.min(Ꮬ_目)/3
	σ = 2*np.sqrt(np.max(ⵙ_目))

	ⵙ_Kᵢﺯ = np.exp(-(ⵙ_目)/(2*σ*σ))
	Ꮬ_Kᵢﺯ = np.exp(-(Ꮬ_目)/(2*σ*σ))

	ⵙ_目 = np.sqrt(ⵙ_目)
	Ꮬ_目 = np.sqrt(Ꮬ_目)

	print('Results : ')
	print('σ : %.4E'%σ)
	print("db['σᵦ'] : %.4E\n"%db['σᵦ'])

	print("Within Cluster stats")
	print("Sample size : %d"%len(ⵙ_目))
	print("max : %.4E"%np.max(ⵙ_目))
	print("min : %.4E"%np.min(ⵙ_目))
	print("mean : %.4E"%np.mean(ⵙ_目))
	print("std : %.4E"%np.std(ⵙ_目))
	print("---------------------")
	print("max similarity: %.4E"%np.max(ⵙ_Kᵢﺯ))
	print("min similarity: %.4E"%np.min(ⵙ_Kᵢﺯ))
	print("mean similarity : %.4E"%np.mean(ⵙ_Kᵢﺯ))
	print("std similarity : %.4E\n\n"%np.std(ⵙ_Kᵢﺯ))


	print("Between Cluster stats")
	print("Sample size : %d"%len(Ꮬ_目))
	print("max : %.4E"%np.max(Ꮬ_目))
	print("min : %.4E"%np.min(Ꮬ_目))
	print("mean : %.4E"%np.mean(Ꮬ_目))
	print("std : %.4E"%np.std(Ꮬ_目))
	print("---------------------")
	print("max similarity: %.4E"%np.max(Ꮬ_Kᵢﺯ))
	print("min similarity: %.4E"%np.min(Ꮬ_Kᵢﺯ))
	print("mean similarity : %.4E"%np.mean(Ꮬ_Kᵢﺯ))
	print("std similarity : %.4E"%np.std(Ꮬ_Kᵢﺯ))

	
	γ = 1.0/(2*db['σₐ']*db['σₐ'])
	Kₐ = sklearn.metrics.pairwise.rbf_kernel(db['X'], gamma=γ)
	γ = 1.0/(2*σ*σ)
	Kⲃ = sklearn.metrics.pairwise.rbf_kernel(db['X'].dot(ƝՃ), gamma=γ)

	plt.subplot(121)
	plt.imshow(1-Kₐ, cmap='Blues_r', interpolation='nearest') #cmap options = viridis,Blues_r,hot
	plt.title('Initial Kernel')
	plt.subplot(122)
	plt.imshow(1-Kⲃ, cmap='Blues_r', interpolation='nearest') #cmap options = viridis,Blues_r,hot
	plt.title('Kernel Matrix After 1 Layer')
	plt.show()


def study_ƝՃ_為ⵙ(db, ƝՃ):
	#	seeing same class
	X = db['X']
	Y = db['Y']
	ð = X.shape[1]
	ȿ_目 = []

	Ł_目 = np.unique(Y)
	for i in Ł_目:
		indices = np.where(Y == i)
		ϰ = X[indices, :][0]
		ɲ = ϰ.shape[0]
	
		for i in np.arange(ɲ):
			for j in np.arange(ɲ):
				xᵢ = np.reshape(ϰ[i,:], (ð, 1))
				xﺯ = np.reshape(ϰ[j,:], (ð, 1))
				Δx = xᵢ - xﺯ
	
				Aᵢﺯ = Δx.dot(Δx.T)

				ȿ = np.trace(ƝՃ.T.dot( Aᵢﺯ ).dot( ƝՃ ))
				ȿ_目.append(ȿ)

	ȿ_目 = np.array(ȿ_目)
	return ȿ_目

#	E目 = np.exp(-(ȿ_目*ȿ_目)/(2*σ*σ))
#	print("---------------------")
#	print("max similarity: %.4E"%np.max(E目))
#	print("min similarity: %.4E"%np.min(E目))
#	print("mean similarity : %.4E"%np.mean(E目))
#	print("std similarity : %.4E"%np.std(E目))
#	print('\n\n')

	#plt.hist(ȿ_目, normed=True, bins=30)
	#plt.show()
	#import pdb; pdb.set_trace()


def study_ƝՃ_為Ꮬ(db, ƝՃ):
	#	seeing different class
	X = db['X']
	Y = db['Y']
	ð = X.shape[1]

	Ł_目_i = np.unique(Y)
	Ł_目_j = np.unique(Y)
	ȿ_目 = []

	for i in Ł_目_i:
		for j in Ł_目_j:
			if i != j:
				indices_i = np.where(Y == i)
				indices_j = np.where(Y == j)

				X_A = X[indices_i, :][0]
				X_B = X[indices_j, :][0]

				na = X_A.shape[0]
				nb = X_B.shape[0]

				for α in np.arange(na):
					for β in np.arange(nb):
						xᵢ = np.reshape(X_A[α,:] , (ð, 1))
						xﺯ = np.reshape(X_B[β,:] , (ð, 1))
						Δx = xᵢ - xﺯ
						Aᵢﺯ = Δx.dot(Δx.T)

						ȿ = np.trace(ƝՃ.T.dot( Aᵢﺯ ).dot( ƝՃ ))	
						ȿ_目.append(ȿ)

	ȿ_目 = np.array(ȿ_目)
	return ȿ_目

