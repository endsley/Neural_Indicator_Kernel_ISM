

from src.tools.split_10_fold import *
from src.tests.run_single_mlp_via_sgd import *
from src.tests.run_single_DeepGP import *
from src.tests.run_single_NTK import *
from src.tests.run_single_arcos import *
from src.tests.run_single_ikpca import *

from src.tools.collect_HSIC_CE_MSE_results import collect_HSIC_CE_MSE_results


def aggregate_all_sgd_results(data_name, loss_function):
	pth_name = './results/' + data_name + '/' + data_name + '_'

	Ð = {}
	Ð2 = {}
	ℓ_info = ''
	results_aggregate = ''
	for ᘐ in range(1,11):
		file_name = pth_name + str(ᘐ) + '/' + 'sgd_' + loss_function + '_relu.txt'
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

		#ℓ_info += data_name + '_' + str(ᘐ) + ' : '
		#for ł in results_per_line[4:]:
		#	ℓ_info += 'Layer ' + ł.strip() + ' , '
		#ℓ_info += '\n'


	for ᘔ, Ⴥ in Ð.items():
		Ð2[ᘔ] = '%.3f ± %.3f'%(np.mean(np.array(Ⴥ)),np.std(np.array(Ⴥ)))
		
	pth_name = './results/' + data_name + '/sgd_10_fold_' + loss_function + '.txt'
	fout = open(pth_name, 'w')
	print('Running ' + data_name + ' , with ' + loss_function)
	for ᘔ, Ⴥ in Ð2.items():
		output = '%-15s%-10s'%(ᘔ,Ⴥ)
		print('   ' + output)
		fout.write(output + '\n')

	print('\n\n' + title_str + results_aggregate)
	fout.write('\n' + title_str + results_aggregate)
	fout.close()


def run_10_fold_via_sgd(data_name, KN, loss_function):
	gen_10_fold_data(data_name, data_path='./data/')	
	
	pth_name = data_name + '/' + data_name + '_'

	for ᘐ in range(1,11):
		file_name = pth_name + str(ᘐ)
		if loss_function == 'DeepGP':
			run_single_DeepGP(file_name, KN, loss_function)
		elif loss_function == 'ikpca':
			print('----------------')
			run_single_ikpca(file_name, KN, loss_function)
		elif loss_function == 'arcos':
			run_single_arcos(file_name, KN, loss_function)
		elif loss_function == 'NTK':
			run_single_NTK(file_name, KN, loss_function)
		else:
			run_single_mlp_via_sgd(file_name, KN, loss_function)

	aggregate_all_sgd_results(data_name, loss_function)
	collect_HSIC_CE_MSE_results(data_name)
