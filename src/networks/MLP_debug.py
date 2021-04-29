

import sklearn.metrics
from sklearn.metrics import accuracy_score
from ..tools.hsic import ℍ
from ..tools.distances import *
from ..tools.file_write import *

def record_results_to_txt(db):
	if not db['network_info']['knet'].db['record_internal']: return
	netInfo = db['network_info']

	X = db['train_data'].X_Var
	Y = db['network_info']['knet'].db['Y']
	Yₒ = db['network_info']['knet'].db['Yₒ']
	Ȳ = db['mlp'].forward_with_end_cap(X)


	loss_method = getattr(db['mlp'], db['loss_function'])

	train_Łabel = db['mlp'].fit(db['train_data'].X_Var, db['loss_function'])
	test_Łabel = db['mlp'].fit(db['test_data'].X_Var, db['loss_function'])



	db['train_CE'] = db['mlp'].CE_loss(db['train_data'].X_Var, db['train_data'].Y_Var, None)
	db['test_CE'] = db['mlp'].CE_loss(db['test_data'].X_Var, db['test_data'].Y_Var, None)
	db['train_MSE'] = db['mlp'].MSE_loss(db['train_data'].X_Var, db['train_data'].Y_Var, None)
	db['test_MSE'] = db['mlp'].MSE_loss(db['test_data'].X_Var, db['test_data'].Y_Var, None)


#	train_out = db['mlp'].forward(db['train_data'].X_Var).detach().numpy()
#	test_out = db['mlp'].forward(db['test_data'].X_Var).detach().numpy()
#	import pdb; pdb.set_trace()
#
#	db['train_MSE'] = MSE(train_out, db['train_data'].Y)
#	db['test_MSE'] = MSE(test_out, db['test_data'].Y)




	db['train_acc'] = accuracy_score(train_Łabel, db['train_data'].Y)
	db['test_acc'] = accuracy_score(test_Łabel, db['test_data'].Y)

	np.hstack((Ȳ,Yₒ))
	
	if db['loss_function'] == 'MSE_loss': hsic = ℍ(Ȳ,Yₒ, Kᵪ_type='Gaussian')
	else: hsic = ℍ(Ȳ,Yₒ)

	ȿ = sklearn.metrics.silhouette_score(Ȳ, Y)	

	[mean_intra_cos_sim, σ_intra_cos_sim] = mean_intra_cluster_cosine_similarity(Ȳ,Y)
	[mean_inter_cos_sim, σ_inter_cos_sim] = mean_inter_cluster_cosine_similarity(Ȳ,Y)
	Ճ = mean_inter_cos_sim/mean_intra_cos_sim

	Ł = ('Train_Acc','Test_Acc','Time(s)', 'HSIC', 'Silhouette', 'MSE', 'CE', 'CSR')
	ᘐ = (db['train_acc'], db['test_acc'], db['ΔTime'], hsic, ȿ, db['train_MSE'], db['train_CE'], Ճ)

	Ł_ʆ = ("%-10s\t%-10s\t%-10s\t%-10s\t%-10s\t%-10s\t%-10s\t%-10s"%Ł)
	Ꮙ_ʆ = ("%-10.3f\t%-10.3f\t%-10.3f\t%-10.3f\t%-10.3f\t%-10.3f\t%-10.3f\t%-10.3f"%ᘐ)
	ʆ = Ł_ʆ + '\n' + Ꮙ_ʆ + '\n\n'

	ʆ += '%-30s%d\n'%('Data Dimension :', netInfo['d'])
	ʆ += '%-30s%d\n'%('Num of Classes :', netInfo['c'])
	ʆ += '%-30s%d\n'%('Network Width :', netInfo['width'])
	ʆ += '%-30s%s\n'%('Activation Function :', netInfo['activation_function'])
	ʆ += '%-30s%s\n'%('Loss Function :', db['loss_function'])
	ʆ += '%-30s%d\n'%('Batch size :', db['batch_size'])


	
	print('\n' + ʆ) 
	
	write_train_results(db, ʆ, result_file_name='sgd_' + db['loss_function'] + '_' + db['mlp'].Φ_type + '.txt')



