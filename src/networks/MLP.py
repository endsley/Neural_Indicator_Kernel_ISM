#!/usr/bin/env python

import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import time 

class MLP(torch.nn.Module):
	def __init__(self, network_info, learning_rate=0.001):
		super(MLP, self).__init__()
		self.learning_rate = learning_rate
		self.Φ_type = Φ_type = network_info['activation_function']
		self.loss_function = network_info['loss_function']

		start_id = 2
		exec('self.l1 = torch.nn.Linear(' + str(network_info['d']) + ', ' + str(network_info['width']) + ' , bias=True)')
		exec('self.l1.activation = "' + Φ_type + '"')		#softmax, relu, tanh, sigmoid, none

		if not network_info['knet'].db['start_from_RKHS']: 
			network_info['network_structure'].pop(0)

		for ᘔ, ℓ in enumerate(network_info['network_structure'], start=start_id):
			exec('self.l' + str(ᘔ) + ' = torch.nn.Linear(' + str(network_info['width']) + ', ' + str(network_info['width']) + ' , bias=True)')
			exec('self.l' + str(ᘔ) + '.activation = "' + Φ_type + '"')		#softmax, relu, tanh, sigmoid, none

		exec('self.l' + str(ᘔ+1) + ' = torch.nn.Linear(' + str(network_info['width']) + ', ' + str(network_info['c']) + ' , bias=True)')
		exec('self.l' + str(ᘔ+1) + '.activation = "none"')		#softmax, relu, tanh, sigmoid, none


		self.initialize_network()
		self.output_network()

	def initialize_network(self):
		for param in self.parameters():
			if(len(param.data.numpy().shape)) > 1:
				torch.nn.init.kaiming_normal_(param.data , a=0, mode='fan_in')	
			else:
				pass
				#param.data = torch.zeros(param.data.size())

		self.num_of_linear_layers = 0
		for m in self.children():
			if type(m) == torch.nn.Linear:
				self.num_of_linear_layers += 1

		print('Number of layers : %d'%self.num_of_linear_layers)

	def output_network(self):
		print('\tConstructing Kernel Net')
		for i in self.children():
			try:
				print('\t\t%s , %s'%(i,i.activation))
			except:
				print('\t\t%s '%(i))

	def get_optimizer(self):
		return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

	def forward_with_softmax(self, X):
		y_out = self.forward(X)
		y_out = F.softmax(y_out)
		y_out = y_out.detach().numpy()
		return y_out

	def forward_with_sum(self, X):
		y_out = self.forward(X)
		y_out = torch.sum(y_out, dim=1)
		y_out = y_out.detach().numpy()
		if len(y_out.shape) == 1: y_out = np.reshape(y_out, (y_out.shape[0], 1))
		return y_out

	def forward_with_end_cap(self, X):
		if self.loss_function == 'CE_loss':
			return self.forward_with_softmax(X)
		else:
			return self.forward_with_sum(X)

	def fit(self, X, error_type):
		y_out = self.forward(X)

		if error_type == 'CE_loss':
			y_out = F.softmax(y_out)
			y_out = y_out.detach().numpy()
			Łabel = np.argmax(y_out, axis=1)
		else:
			y_out = torch.sum(y_out, dim=1)
			y_out = y_out.detach().numpy()
			Łabel = np.round(y_out)
		return Łabel.astype(int)

	def CE_loss(self, x, y_true, indices):
		y_true = y_true.long()
		y_pred = self.forward(x)

		loss = torch.nn.functional.cross_entropy(y_pred, y_true)
		return loss

	def MSE_loss(self, x, y_true, indices):
		y_pred = self.forward(x)
		y_out = torch.sum(y_pred, dim=1)
		loss = torch.nn.functional.mse_loss(y_out,y_true)
		return loss

	def forward(self, y0):
		for m, layer in enumerate(self.children(),1):
			if m == self.num_of_linear_layers:
				cmd = 'self.y_pred = self.l' + str(m) + '(y' + str(m-1) + ')'
				#print(cmd)
				exec(cmd)
				return self.y_pred
			else:
				var = 'y' + str(m)
				cmd = var + ' = F.' + self.Φ_type + '(self.l' + str(m) + '(y' + str(m-1) + '))'
				#print(cmd)
				exec(cmd)


