

from src.tools.path_tools import path_list_exists
from src.tools.file_write import *

def extract_title(data_name, path):
	fin = open(path, 'r')
	info = fin.readlines()
	fin.close()

	title_line = '%-20s'%''
	for i in range(9):
		title_line += '%-20s'%(info[i][0:15].strip())

	return title_line + '\n'

def extract_info(data_name, path):
	fin = open(path, 'r')
	info = fin.readlines()
	fin.close()

	line_info = '%-20s'%data_name
	for i in range(9):
		line_info += '%-20s'%(info[i][15:].strip())
		
	return line_info + '\n'


def collect_HSIC_CE_MSE_results(data_name):
	db = {}
	db['data_name'] = data_name

	ism_path = './results/' + data_name + '/ism_10_fold.txt' 
	ce_path = './results/' + data_name + '/sgd_10_fold_CE_loss.txt' 
	mse_path = './results/' + data_name + '/sgd_10_fold_MSE_loss.txt' 
	path_list = [ism_path, ce_path, mse_path]

	if not path_list_exists(path_list): return

	output_content = extract_title(data_name, ism_path)
	output_content +=  extract_info(data_name + '/HSIC', ism_path)
	output_content +=  extract_info(data_name + '/CE', ce_path)
	output_content +=  extract_info(data_name + '/MSE', mse_path)

	print(output_content)
	write_train_results(db, output_content, result_file_name='HSIC_MSE_CE_results.txt')
