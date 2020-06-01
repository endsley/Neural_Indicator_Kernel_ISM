

from .path_tools import *

def write_train_results(db, ʆ, result_file_name='result.txt'):
	pth = './results/' + db['data_name']
	ensure_path_exists(pth)
	fin = open(pth + '/' + result_file_name, 'w')
	fin.write(ʆ)
	fin.close()

