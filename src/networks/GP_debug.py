

import sklearn.metrics
from sklearn.metrics import accuracy_score
from ..tools.hsic import ℍ
from ..tools.distances import *
from ..tools.file_write import *
from src.dknet.Deep_Kernel_Gaussian_Process import *

def record_results_to_txt(db):
	if not db['network_info']['knet'].db['record_internal']: return
	netInfo = db['network_info']

	#X = db['train_data'].X
	#Y = db['train_data'].Y
	#X_test = db['test_data'].X 
	#Y_test = db['test_data'].Y

	#import pdb; pdb.set_trace()
	#mu, stdt = db['dGP'].predict(X,return_std=True)
	#train_Łabel = np.rint(mu)

	#mu, stdt = db['dGP'].predict(X_test,return_std=True)
	#test_Łabel = np.rint(mu)

	#	Set these to empty
	db['train_CE'] = 0
	db['test_CE'] = 0
	db['train_MSE'] = 0
	db['test_MSE'] = 0
	hsic = 0
	ȿ = 0
	Ճ = 0

	#db['train_acc'] = accuracy_score(train_Łabel, db['train_data'].Y)
	#db['test_acc'] = accuracy_score(test_Łabel, db['test_data'].Y)

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
	
	write_train_results(db, ʆ, result_file_name='sgd_' + db['loss_function'] + '_relu' + '.txt')



