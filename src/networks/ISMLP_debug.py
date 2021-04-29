

import sklearn.metrics
import time
import numpy as np
from ..tools.hsic import ℍ
from ..tools.distances import *
from ..tools.plot_lib import *
from ..tools.file_write import *
from ..libs.ISM_supervised_DR.src.tools.classifiers import *

def record_each_ℓ(Ꮬ, new_ℓ):
	db = Ꮬ.db 
	if not db['record_internal']: return

	ℓᴼᵁᵀ = new_ℓ.ℓᴼᵁᵀ
	zℓᴼᵁᵀ = XWₐ = new_ℓ.XWₐ
	K = ℓᴼᵁᵀ.dot(ℓᴼᵁᵀ.T)

	[mean_intra_cos_sim, σ_intra_cos_sim] = mean_intra_cluster_cosine_similarity(ℓᴼᵁᵀ,db['Y'])
	[mean_inter_cos_sim, σ_inter_cos_sim] = mean_inter_cluster_cosine_similarity(ℓᴼᵁᵀ,db['Y'])
	mse = MSE(XWₐ,db['Y'])
	CE = Get_Cross_Entropy(ℓᴼᵁᵀ,db['Y'])


	layer_id = len(Ꮬ.ℓ_目)
	Ꮬ.Ɗ['#_of_layers'] = layer_id
	db['time_目'].append(db['ΔTime'])

	Ꮬ.Ɗ[layer_id] = {}
	Ꮬ.Ɗ[layer_id]['Wₐ_shape'] = new_ℓ.Wₐ.shape
	Ꮬ.Ɗ[layer_id]['ℓ_width'] = new_ℓ.Wₐ.shape[1]
	Ꮬ.Ɗ[layer_id]['HSIC'] = ℍ(ℓᴼᵁᵀ, db['Yₒ'])		
	Ꮬ.Ɗ[layer_id]['silhouette_score'] = db['ȿ']		
	Ꮬ.Ɗ[layer_id]['max〡xᴵ-xᴶ〡_đ'] = np.max(sklearn.metrics.pairwise.pairwise_distances(ℓᴼᵁᵀ))
	Ꮬ.Ɗ[layer_id]['mean_intra_cos_sim'] = mean_intra_cos_sim
	Ꮬ.Ɗ[layer_id]['mean_inter_cos_sim'] = mean_inter_cos_sim
	Ꮬ.Ɗ[layer_id]['mse'] = mse
	Ꮬ.Ɗ[layer_id]['CE'] = CE
	Ꮬ.Ɗ[layer_id]['zℓᴼᵁᵀ'] = zℓᴼᵁᵀ

	plot_heatMap(db, K, layer_id)

def print_each_ℓ_ℹ(Ꮬ, new_ℓ):
	db = Ꮬ.db 
	if not db['print_internal_at_each_stage']: return

	XWₐ = new_ℓ.XWₐ
	Wₐ = new_ℓ.Wₐ
	Wᵦ = new_ℓ.Wᵦ
	K = new_ℓ.Φᵪ.dot(new_ℓ.Φᵪ.T)
	Y = db['Y']
	layer_id = len(Ꮬ.ℓ_目)
	
	labels = np.unique(db['Y'])
	Φᵪ_1 = new_ℓ.Φᵪ[Y == labels[0], 0:20]
	Φᵪ_2 = new_ℓ.Φᵪ[Y == labels[1], 0:20]
	K_same = (Φᵪ_1.dot(Φᵪ_1.T))[0:10,0:20]
	K_diff = (Φᵪ_1.dot(Φᵪ_2.T))[0:10,0:20]

	#ẙᑕY = db['Y'].reshape(1,db['N'])[0, 0:20]
	#Ƙ = np.vstack((ẙᑕY, K[0:10,0:20]))

	#	Compute HSIC
	ℓᴼᵁᵀ = new_ℓ.ℓᴼᵁᵀ
	Yₒ = db['Yₒ']
	ɦ = ℍ(ℓᴼᵁᵀ, Yₒ)		

	max_ij_đ = np.max(sklearn.metrics.pairwise.pairwise_distances(ℓᴼᵁᵀ))
	mean_intra_đ = mean_intra_cluster_pairwise_distances(ℓᴼᵁᵀ,db['Y'])
	[mean_intra_cos_sim, σ_intra_cos_sim] = mean_intra_cluster_cosine_similarity(ℓᴼᵁᵀ,db['Y'])
	[mean_inter_cos_sim, σ_inter_cos_sim] = mean_inter_cluster_cosine_similarity(ℓᴼᵁᵀ,db['Y'])
	mse = MSE(XWₐ,db['Y'])
	CE = Get_Cross_Entropy(ℓᴼᵁᵀ,db['Y'])

	print('\tTime(s) : %.3f'%db['ΔTime'])
	print('\tCurrent HSIC : %.3f'%ɦ)
	print('\tMSE %.3f'%mse)
	print('\tCE %.3f'%CE)
	print('\tmax ⅆ %.3f'%max_ij_đ)
	print('\tmax mean intra ⅆ %.3f'%mean_intra_đ)
	print('\tMean intra cos similarity %.3f 土 %3f'%(mean_intra_cos_sim, σ_intra_cos_sim))
	print('\tMean inter cos similarity %.3f 土 %3f'%(mean_inter_cos_sim, σ_inter_cos_sim))
	print('\tσ used	: %.3f'%new_ℓ.σ)
	print('\tWₐ	: ',str(Wₐ.shape))
	print('\tWᵦ	: ',str(Wᵦ.shape))
	print('\tRFF # of samples : %d'%db['RFF_#_samples'])
	print('\tSilhouette score	: %.3f'%db['ȿ'])
	print('\tLayer type : %s'%db['layer'].__name__)
	print('\tKernel of Sampe Class	: \n\t',str(K_same).replace('\n','\n\t'))
	print('\tKernel of Different Classes	: \n\t',str(K_diff).replace('\n','\n\t'))

	if XWₐ.shape[1] == 2: scatter(db, XWₐ, db['Y'], layer_id)


def record_results_to_txt(train_acc, test_acc, knet):
	if not knet.db['record_internal']: return

	mlp = knet.db['network']
	Ɗ = knet.db['network'].Ɗ
	Ꮬ = Ɗ[Ɗ['#_of_layers']]
	
	CSR = Ꮬ['mean_inter_cos_sim']/Ꮬ['mean_intra_cos_sim']	# cosine similarity ratio
	run_time = np.sum(np.array(knet.db['time_目']))


	Ł = ('Train_Acc','Test_Acc','Time(s)', 'HSIC', 'MSE', 'CE', 'CSR', 'Silhouette')
	ᘐ = (train_acc, test_acc, run_time, Ꮬ['HSIC'], Ꮬ['mse'], Ꮬ['CE'], CSR, Ꮬ['silhouette_score'])

	Ł_ʆ = ("%-10s\t%-10s\t%-10s\t%-10s\t%-10s\t%-10s\t%-10s\t%-10s"%Ł)
	Ꮙ_ʆ = ("%-10.3f\t%-10.3f\t%-10.3f\t%-10.3f\t%-10.3f\t%-10.3f\t%-10.3f\t%-10.3f"%ᘐ)



	ʆ = Ł_ʆ + '\n' + Ꮙ_ʆ + '\n\nLayer Weight Dimension : \n'
	
	for i, ℓ in enumerate(mlp.ℓ_目):
		ʆ = ʆ + ("\t%d : %s\n"%(i, str(ℓ.Wₐ.shape) ))

	write_train_results(knet.db['network'].db, ʆ)
	print(ʆ) 


def record_final_ℓ(Ꮬ, xᴵᴺ):
	db = Ꮬ.db 
	if not db['record_internal']: return

	final_ℓ = Ꮬ.final_ℓ
	Ł_目 = Ꮬ.Ł_目
	ⵙ_田 = Ꮬ.ⵙ_田

	db['time_目'].append(db['ΔTime'])
	num_layer = len(Ꮬ.ℓ_目)


	Y = db['Y']
	n = len(Y)

	count = np.reshape(np.arange(1,n+1), (n,1))
	Y = np.reshape(Y,(n,1))
	Φ = final_ℓ.apply_layer(xᴵᴺ)

	Ł = Ꮬ.get_label(Φ)
	Ł = np.reshape(Ł, (n,1))
	out_田 = np.hstack((count, Ł, Y,Φ))
	train_acc = sklearn.metrics.accuracy_score(Ł, db['Y'])

	Ꮬ.Ɗ['ⵙ_田'] = ⵙ_田
	Ꮬ.Ɗ['train_acc'] = train_acc


	HSIC_目 = []
	ȿ_目 = []
	mse_目 = []
	ce_目 = []
	cos_目 = []
	Λ_目 = []

	for i in np.arange(1,num_layer+1):
		ȿ_目.append( Ꮬ.Ɗ[i]['silhouette_score'] )
		HSIC_目.append(Ꮬ.Ɗ[i]['HSIC'])
		cos_目.append( Ꮬ.Ɗ[i]['mean_inter_cos_sim']/Ꮬ.Ɗ[i]['mean_intra_cos_sim'] )
		mse_目.append( Ꮬ.Ɗ[i]['mse'] )
		ce_目.append( Ꮬ.Ɗ[i]['CE'] )
		Λ_目.append( Ꮬ.Ɗ[i]['ℓ_width'] )
		

	mse_目 = mse_目/np.max(mse_目)
	Λ_目 = Λ_目/np.max(Λ_目)
	#ce_目 = ce_目/np.max(ce_目)
	#score_目 = [cos_目, ȿ_目, HSIC_目, mse_目, ce_目, Λ_目]
	#Ł_目 = ['CS','Ŝ','HSIC','MSE', 'CE', 'Λ']
	score_目 = [cos_目, ȿ_目, HSIC_目, mse_目, ce_目]
	Ł_目 = ['CS','$\mathscr{S}$','HSIC','MSE', 'CE']

	title = 'Key Metrics : %s'%db['data_name'].split('/')[0]
	xlabel = 'Network Layers'
	ylabel = 'Score'
	
	plot_目(db, score_目, Ł_目, title, xlabel, 0, ylabel)

	out_path = './results/' + db['data_name'] + '/mlp.pk'
	Ꮬ.save_network(out_path)

def print_final_ℓ(MLP_obj, xᴵᴺ):
	db = MLP_obj.db 
	if not db['print_internal_at_each_stage']: return

	final_ℓ = MLP_obj.final_ℓ
	Ł_目 = MLP_obj.Ł_目
	ⵙ_田 = MLP_obj.ⵙ_田

	Y = db['Y']
	n = len(Y)

	count = np.reshape(np.arange(1,n+1), (n,1))
	Y = np.reshape(Y,(n,1))
	Φ = final_ℓ.apply_layer(xᴵᴺ)

	Ł = MLP_obj.get_label(Φ)
	Ł = np.reshape(Ł, (n,1))
	out_田 = np.hstack((count, Ł, Y,Φ))
	train_acc = sklearn.metrics.accuracy_score(Ł, db['Y'])

	#print(out_田)
	print(ⵙ_田)
	print('train accuracy : %.3f'%train_acc)
	print('Run Time : %.5f'%db['ΔTime'])
