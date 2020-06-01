
import numpy as np
from src.tools.file_write import *

def record_label_stats(data_name, Y, Łabel, Ł_test):
	db = {}
	db['data_name'] = data_name

	labels = np.unique(Y)
	ʆ = '%-10s'%'Train'

	for i in labels:
		pc = np.sum(Łabel == i)/len(Łabel)
		ʆ += 'label %d : %-20.3f'%(i,pc)
	ʆ += '\n%-10s'%'Test'

	for i in labels:
		pc = np.sum(Ł_test == i)/len(Ł_test)
		ʆ += 'label %d : %-20.3f'%(i,pc)

	write_train_results(db, ʆ, result_file_name='label_stats.txt')
	print(ʆ) 

