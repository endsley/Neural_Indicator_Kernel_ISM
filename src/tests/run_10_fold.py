
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from src.networks.ISMLP_debug import *
from src.tools.split_10_fold import *
from src.tools.collect_HSIC_CE_MSE_results import collect_HSIC_CE_MSE_results
from src.tests.combine_kernel_img import combine_kernel_img
from src.tools.label_stats import *


def aggregate_all_ism_results(data_name):
	pth_name = './results/' + data_name + '/' + data_name + '_'

	Ð = {}
	Ð2 = {}
	ℓ_info = ''
	results_aggregate = ''
	for ᘐ in range(1,11):
		file_name = pth_name + str(ᘐ) + '/result.txt'
		fin = open(file_name)
		results_per_line = fin.readlines()
		fin.close()

		results_title = results_per_line[0].split()
		results_vals = results_per_line[1].split()
		title_str = results_per_line[0]
		results_aggregate += results_per_line[1].strip() + '\t: Layer ' + str(ᘐ) + '\n'

		for ᘔ, Ⴥ in enumerate(results_title):
			if Ⴥ not in Ð:
				Ð[Ⴥ] = []

			Ð[Ⴥ].append(float(results_vals[ᘔ]))

		ℓ_info += data_name + '_' + str(ᘐ) + ' : '
		for ł in results_per_line[4:]:
			ℓ_info += 'Layer ' + ł.strip() + ' , '
		ℓ_info += '\n'


	for ᘔ, Ⴥ in Ð.items():
		Ð2[ᘔ] = '%.2f ± %.2f'%(np.mean(np.array(Ⴥ)),np.std(np.array(Ⴥ)))
		
	pth_name = './results/' + data_name + '/ism_10_fold.txt'
	fout = open(pth_name, 'w')
	for ᘔ, Ⴥ in Ð2.items():
		output = '%-15s%-10s'%(ᘔ,Ⴥ)
		print(output)
		fout.write(output + '\n')

	print('\n'+ ℓ_info)
	print(title_str + '\n'+ results_aggregate)

	fout.write('\n'+ ℓ_info)
	fout.write('\n\n' + title_str + results_aggregate)
	fout.close()

def run_10_fold(data_name, KN):
	gen_10_fold_data(data_name, data_path='./data/')	
	
	pth_name = data_name + '/' + data_name + '_'

	for ᘐ in range(1,11):
		file_name = pth_name + str(ᘐ)

		X = np.loadtxt('data/' + file_name + '.csv', delimiter=',', dtype=np.float64)			
		Y = np.loadtxt('data/' + file_name + '_label.csv', delimiter=',', dtype=np.int32)			
		X_test = np.loadtxt('data/' + file_name + '_test.csv', delimiter=',', dtype=np.float64)			
		Y_test = np.loadtxt('data/' + file_name + '_label_test.csv', delimiter=',', dtype=np.int32)			
	
		X = preprocessing.scale(X)
		X_test = preprocessing.scale(X_test)
	
		knet = KN(X,Y, file_name)	#q if not set, it is automatically set to 80% of data variance by PCA
		knet.train()

		Łabel = knet.fit(X)
		train_acc = accuracy_score(Łabel, Y)
	
		Ł_test = knet.fit(X_test)
		test_acc = accuracy_score(Ł_test, Y_test)
	
		record_results_to_txt(train_acc, test_acc, knet)
		combine_kernel_img(file_name)
		#record_label_stats(data_name, Y, Łabel, Ł_test)

	aggregate_all_ism_results(data_name)
	collect_HSIC_CE_MSE_results(data_name)
