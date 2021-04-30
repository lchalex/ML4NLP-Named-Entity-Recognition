from copy import copy, deepcopy
import os
import os.path as osp
import time
import pdb

import torch
import torch.nn as nn

WEIGHTS_ROOT = '../weights'

class LSTMNet(nn.Module):
	def __init__(self, embed_size, class_num, layer_n=2, hidden_dim=128):
		super(LSTMNet, self).__init__()
		self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
		self.layer_n = layer_n
		self.hidden_dim = hidden_dim
		self.embedding = nn.Embedding(embed_size, 128)
		self.lstm1 = nn.LSTM(128, hidden_dim, layer_n, batch_first=True, dropout=0.2, bidirectional=True)
		self.conv2 = nn.Conv1d(128, 128, kernel_size=5, stride=1, padding=2)
		self.lstm2 = nn.LSTM(128, hidden_dim, layer_n, batch_first=True, dropout=0.2, bidirectional=True)
		self.fc = nn.Linear(hidden_dim * 2 * 2, class_num)
	
	def forward(self, x):
		x = self.embedding(x)

		x1, h1 = self.lstm1(x)

		x2 = x.permute(0, 2, 1)
		x2 = self.conv2(x2)
		x2 = x2.permute(0, 2, 1)
		x2, h2 = self.lstm2(x2)

		x = torch.cat([x1, x2], 2)
		x = self.fc(x)
		x = x.permute(0, 2, 1)
		return x
        
def train_model(model, dataloaders, learn_rate, regulation, fold, num_epoch=100):
	device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
	data_shape = next(iter(dataloaders['train']))[0].shape
	batch_size = data_shape[0]
	model.to(device)
	
	# Defining loss function and optimizer
	criterion = torch.nn.CrossEntropyLoss().to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate, weight_decay=regulation)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.4)

	print("Starting Training of LSTM model")
	epoch_times = []
	best_acc = 0
	update = 0
	best_model_sd = deepcopy(model.state_dict())
	# Start training loop
	for epoch in range(1, num_epoch + 1):
		print("Epoch {}/{}".format(epoch, num_epoch))
		for phase in ['train', 'valid']:
			start_time = time.clock()
			total_corrects = 0
			avg_loss = 0.
			if phase == 'train':
				model.train()  # Set model to training mode
			else:
				model.eval()   # Set model to evaluate mode

			for x, label in dataloaders[phase]:
				label = label.to(device)
				x = x.to(device)
				optimizer.zero_grad()
				with torch.set_grad_enabled(phase == 'train'):
					out = model(x)
					_, preds = torch.max(out, 1)
					loss = criterion(out, label)

					if phase == 'train':
						loss.backward()
						optimizer.step()
				
				avg_loss += loss.item()
				total_corrects += torch.sum(preds == label.data)

			current_time = time.clock()
			acc = float(total_corrects.double()/128/len(dataloaders[phase].dataset))
			print("\tPhase : {}, Total Loss: {:.5f}, Acc: {:.5f}".format(phase, avg_loss/len(dataloaders[phase]), acc))
			epoch_times.append(current_time-start_time)
			if phase == 'valid':
				if acc > best_acc:
					print("Update best model sd")
					best_model_sd = deepcopy(model.state_dict())
					best_acc = acc
					update = 0
			
				else:
					update -= 1
			
		scheduler.step()
		if update < -5:
			print("No more improvement. Terminated")
			break

	print("Total Training Time: {} seconds".format(str(sum(epoch_times))))
	torch.save(best_model_sd, osp.join(WEIGHTS_ROOT, 'LSTMNet_' + str(round(best_acc, 3)) + "fold" + str(fold) + ".pth"))
	return model

def eval_model(model, dataloaders, checkpoint):
	device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
	data_shape = next(iter(dataloaders['valid']))[0].shape
	batch_size = data_shape[0]
	criterion = torch.nn.CrossEntropyLoss().to(device)
	model.load_state_dict(torch.load(checkpoint))
	model.to(device)
	model.eval()
	total_corrects = 0
	avg_loss = 0.
	for x, label in dataloaders['valid']:
		label = label.to(device)
		x = x.to(device)
		with torch.set_grad_enabled(False):
			out = model(x)
			_, preds = torch.max(out, 1)
			loss = criterion(out, label)

		avg_loss += loss.item()
		total_corrects += torch.sum(preds == label.data)
	
	print("Eval model, Total Loss: {:.5f}, Acc: {:.5f}".format(avg_loss/len(dataloaders['valid']), total_corrects.double()/128/len(dataloaders['valid'].dataset)))
	return 0

def pred_model(model, dataloaders, checkpoints):
	device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
	ensemble_out = None
	for ckpt in checkpoints:
		model.load_state_dict(torch.load(osp.join(WEIGHTS_ROOT, ckpt)))
		model.to(device)
		model.eval()
		single_model_out = None
		for x, _ in dataloaders['test']:
			x = x.to(device)
			with torch.set_grad_enabled(False):
				out = model(x)
				if single_model_out == None:
					single_model_out = out.cpu().detach()
				else:
					single_model_out = torch.cat([single_model_out, out.cpu().detach()], 0)
		
		if ensemble_out == None:
			ensemble_out = single_model_out
		else:
			ensemble_out += single_model_out
	
	_, preds = torch.max(ensemble_out, 1)
	return preds.numpy().astype(int)