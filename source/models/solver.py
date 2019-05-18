import time
import torch
import numpy as np
from torch.autograd import Variable
from utils.basic import timeSince

class solver():

	def __init__(self, model, lmdb, optimizer, scheduler, total_epoch, model_path, last_epoch):

		self.model = model
		print(self.model)

		self.lmdb_train, self.lmdb_test = lmdb
		self.optimizer = optimizer
		self.scheduler = scheduler
		self.total_epoch = total_epoch
		self.model_path = model_path
		self.last_epoch = last_epoch

		self.start = time.time()

	def train_one_epoch(self, ep):
		pass
	def test_one_epoch(self, ep):
		pass

	def forward(self):
		for ep in range(self.total_epoch-self.last_epoch):
			ep = ep+self.last_epoch
			self.train_one_epoch(ep)
			self.test_one_epoch(ep)
		
import pdb
class seq_solver(solver):

	def train_one_epoch(self, ep):
		self.model.train()
		loss_aver = 0
		if self.scheduler is not None:
			self.scheduler.step()
			print('learning_rate: ', self.scheduler.get_lr())	
		for it, sample_batched in enumerate(self.lmdb_train):
			inputs = sample_batched['image'].squeeze(0)
			labels = sample_batched['label'].squeeze(0)

			inputs = Variable(inputs.cuda())
			loss = self.model(inputs, labels)
			self.optimizer.zero_grad()
			loss.backward()
			loss = loss.data.item()
			l2_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(),10)

			if not np.isnan(l2_norm):
				self.optimizer.step()
			else:
				print('l2_norm: ', l2_norm)
				l2_norm = 0

			if it == 0:
				loss_aver = loss
			loss_aver = 0.9*loss_aver+0.1*loss			  
			if it == len(self.lmdb_train)-1:

				correct_count, len_total, pre_total = self.model.loss_layer.result_analysis(it)

				recall = float(correct_count) / len_total
				precision = correct_count / (pre_total+0.000001)

				print('Train: %10s Epoch: %3d it: %6d, loss: %.4f, l2_norm: %.4f, recall: %.4f, precision: %.4f' % 
					(timeSince(self.start), ep, it, loss_aver, l2_norm, recall, precision))

		torch.save({
			'epoch': ep,
			'state_dict': self.model.state_dict(),
			'optimizer' : self.optimizer.state_dict(),
			}, self.model_path.format(ep)) 	


	def test_one_epoch(self, ep):
		self.model.eval()
		loss_aver = 0
		for it, sample_batched in enumerate(self.lmdb_test):
			inputs = sample_batched['image'].squeeze(0)
			labels = sample_batched['label'].squeeze(0)

			inputs = Variable(inputs.cuda())
			loss = self.model(inputs, labels)
			correct_count, len_total, pre_total = self.model.loss_layer.result_analysis(it)

			loss = loss.data.item()
			if it == 0:
				loss_aver = loss
			loss_aver = 0.9*loss_aver+0.1*loss		

			if it == len(self.lmdb_test) -1:
				recall = float(correct_count) / len_total
				precision = correct_count / (pre_total+0.000001)	
				print('Test : %10s Epoch: %3d it: %6d, loss: %.4f, len : %4d, recall: %.4f, precision: %.4f' % 
							(timeSince(self.start), ep, it, loss_aver, len_total, recall, precision))	


