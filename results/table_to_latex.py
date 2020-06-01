#!/usr/bin/env python

column_num = 10

align = ''
for i in range(column_num): align += '|c'


fin = open('all_results.txt','r')
tb = fin.readlines()
fin.close()

tb_text = '\\begin{table}[h]\n'
tb_text += '\\tiny\n'
tb_text += '\\centering\n'
tb_text += '\\setlength{\\tabcolsep}{3.0pt}\n'
tb_text += '\\renewcommand{\\arraystretch}{1.2}\n'
tb_text += '\\begin{tabular}{' + align + '|}\n'
tb_text += '\t\\hline\n'

cl = ['black','black', 'green', 'darkgreen', 'orange', 'blue', 'purple', 'purple', 'brown', 'brown']

for i, Ꮽ in enumerate(tb):
	if len(Ꮽ) == 1: 
		tb_text += '\t\t\\hline \n'
		continue

	#column_width = int(len(Ꮽ)/column_num)
	column_width = 19

	for j in range(column_num):
		stuff = Ꮽ[j*column_width:j*column_width+column_width].strip()
		stuff = stuff.replace('±', '$\\pm$')
		stuff = stuff.replace('_', ' ')

		if i == 0 or j == 0:
			if j == 0:
				tb_text += '\t ' + stuff + ' &\n'
			elif j == (column_num-1):
				tb_text += '\t\t ' + stuff + ' \\\\ \n'
			else:
				tb_text += '\t\t ' + stuff + ' &\n'
		else:
			if j == 0:
				tb_text += '\t \\' + cl[j] + '{' + stuff + '} &\n'
			elif j == (column_num-1):
				tb_text += '\t\t \\' + cl[j] + '{' + stuff + '} \\\\ \n'
			else:
				tb_text += '\t\t \\' + cl[j] + '{' + stuff + '} &\n'

tb_text += '\t\\hline\n'
tb_text += '\\end{tabular}\n'
tb_text += '\\end{table}\n'
print(tb_text)
