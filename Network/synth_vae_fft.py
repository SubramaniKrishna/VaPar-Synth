"""
Main code to generate audio from a VAE by training it on a parametric model
"""

# Dependencies
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as pyp
import numpy as np
from datetime import datetime
import os
# Importing the dataloader
import sys
from dataset import *
# Importing our model
from vae_krishna import cVAE_synth
from time import time


# Defining the VAE loss function as mentioned in the original VAE paper(https://arxiv.org/pdf/1312.6114.pdf)
def loss_fn(recon_x, x, mean, var, beta = 1):
	MSE = torch.nn.functional.mse_loss(recon_x, x, reduction='sum')
	KLD = -0.5 * torch.sum(1 + var - mean.pow(2) - var.exp())
	return ((MSE/x.size(0)) + beta*KLD), MSE, KLD

# Setting the device to cuda if GPU is available, else to CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'

# Load data into dataloader
batch_size = 512

# Here, provide the path to the pickle dump obtained as output of the parametric modeling
datafile = '../Parametric/Test_octave_4_Spectrum.txt'
# Here, you can select the MIDI pitches you want to train the network on (Should be according to the octave you choose, octave 4 - MIDI 60-71)
pitches = [60,61,70,71]

# This loads the data from the pickle dump into a dataloader. You can choose the total number of frames to train as well here.
dataset = SMSynthDataset(filename = datafile, num_frames = 1000, pitches = pitches)

# Split data into train and test
train_size = int(0.9 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
data_loader_train = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, pin_memory = True, num_workers = 0)
data_loader_test = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, pin_memory = True, num_workers = 0)
len_train = len(train_dataset)
len_test = len(test_dataset)


# Starting the training
num_epochs = 2000

# Define the learning rate
learning_rate = 1.0e-3

# Here, you can provide the hyperparameters {1) beta(controls the relative weighting in the loss function)} and {2) latent space dimension}
# Both are lists so you can iterate over all pairs
beta_list = [0.1]
latent_dims_list = [32]

# Directory to store the PyTorch .pth file (Which contains the network weights)
dir_pth_save = './CVAEfft/'
try: 
    os.makedirs(dir_pth_save, exist_ok = True) 
    print("Directory '%s' created successfully" %dir_pth_save) 
except OSError as error: 
    print("Directory '%s' exists") 


for beta in beta_list:
	for ld in latent_dims_list:

		# Defining the model architecture
		dim_cc = 513
		flag_cond = True
		layer_dims_enc = [513,256,128]
		latent_dims = ld
		layer_dims_dec = [latent_dims,128,256,513]
		num_cond = 1

		now = datetime.now()
		dir_network = dir_pth_save + 'synthmanifold(' + str(now) + ')_' + str([dim_cc, flag_cond, layer_dims_enc, latent_dims, layer_dims_dec, num_cond]) + '_niters_' + str(num_epochs) + '_lr_' + str(learning_rate) + '_beta_' + str(beta) + str(pitches) +  '_net3.pth'

		# Define the model by instantiating an object of the class
		cVAE = cVAE_synth(flag_cond = flag_cond, layer_dims_enc = layer_dims_enc, layer_dims_dec = layer_dims_dec, latent_dims = latent_dims, num_cond = num_cond, device = device).to(device)

		# Defining the optimizer
		optimizer = torch.optim.Adam(cVAE.parameters(), lr=learning_rate)

		# List to store loss
		loss_epoch = []
		MSE_loss = []
		kld_loss =[]
		test_loss = []
		# loss_avg = []
		start = time()

		for epoch in range(num_epochs):
			# bp = 1.0e-9*np.minimum((0.01*epoch),1)
			bp = beta

			tmpsum = 0
			tKLD = 0
			tmptrain = 0
			it = 0
			for iteration, (label, pitch, velocity, x) in enumerate(data_loader_train):
				# it = it + 1
				x, pitch, velocity = x.to(device, non_blocking=True), (pitch.float()/127).to(device, non_blocking=True), (velocity.float()/127).to(device, non_blocking=True)
				# Normalize the input
				# x = x/(abs(x).max(1, keepdim=True)[0])
				x = x/abs(x).max()
				# Ignoring the 0'th cc(only representative of volume/energy)
				# x = x[:,1:]

				c = pitch.view(-1,1)
				x_recon, mu, sigma_covar, z = cVAE(x.float(),c.float())

		        # Calculating the ELBO-Loss
				calc = loss_fn(x_recon.float(), x.float(), mu, sigma_covar, bp)
				loss = calc[0]
				MSE = calc[1].item()
				KLD = calc[2].item()
				tmptrain = tmptrain + loss.item()
				tmpsum = tmpsum + MSE
				tKLD = tKLD + KLD
				# loss_vals.append(loss.item())

				# Performing the optimization by calculating the gradients through backpropagation, and then advancing
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				del loss
				del calc
				del MSE
				# del KLD
			loss_epoch.append(tmptrain/len_train)
			MSE_loss.append(tmpsum/len_train)
			kld_loss.append(tKLD/len_train)
			# MSE.append(MSE)
			print(ld,bp,epoch,'loss = ',loss_epoch[-1])

			# Test Loss Computation
			# tmpsum = 0
			# for iteration, (label, pitch, velocity, x) in enumerate(data_loader_test):
			# 	x, pitch, velocity = x.to(device, non_blocking=False), (pitch.float()/127).to(device, non_blocking=False), (velocity.float()/127).to(device, non_blocking=True)
			# 	# # Normalize the input
			# 	# x = x/(abs(x).max(1, keepdim=True)[0])
			# 	# # Ignoring the 0'th cc(only representative of volume/energy)
			# 	# x = x[:,1:]
			# 	# c = torch.cat((pitch.view(-1,1), velocity.view(-1,1)), dim=-1)
			# 	c = pitch.view(-1,1)
			# 	x_recon, mu, sigma_covar, z = cVAE(x.float(),c.float())

		 #        # Calculating the ELBO-Loss
			# 	calc = loss_fn(x_recon.float(), x.float(), mu, sigma_covar, bp)
			# 	testloss = calc[1].item()
			# 	tmpsum = tmpsum + testloss

			# 	del testloss
			# 	del calc

			# test_loss.append(tmpsum/len_test)


		end = time()
		print('Device Used : ',device)
		print('Total time taken(minutes) : ',(end - start)*1.0/60)
		print('Time per epoch(seconds) :',(end - start)*1.0/num_epochs)


		torch.save(cVAE.state_dict(), dir_network)

	# pyp.figure()
	# pyp.plot(loss_epoch,'r',label = 'Train Loss')
	# # pyp.plot(test_loss,'b',label = 'Test Loss')
	# # pyp.plot(np.array(loss_epoch) - np.array(test_loss),'g',label = 'Difference')
	# # pyp.plot(MSE,'g',label = 'MSE')
	# pyp.plot(kld,'b',label = 'KLD')
	# pyp.title('Train Loss')
	# pyp.xlabel('Number of Epochs')
	# pyp.ylabel('Loss')
	# pyp.legend()
	# pyp.show()

# # # Plotting the Loss function
# # pyp.figure()
# # pyp.plot(loss_epoch,'r',label = 'Train Loss')
# # # pyp.plot(test_loss,'b',label = 'Test Loss')
# # # pyp.plot(np.array(loss_epoch) - np.array(test_loss),'g',label = 'Difference')
# # # pyp.plot(MSE,'g',label = 'MSE')
# # # pyp.plot(kld,'b',label = 'KLD')
# # pyp.title('Train Loss')
# # pyp.xlabel('Number of Epochs')
# # pyp.ylabel('Loss')
# # pyp.legend()
# # # pyp.show()

# pyp.figure()
# pyp.title('MSE Loss')
# # pyp.plot(kld, label = 'KLD Loss')
# pyp.plot(MSE_loss, label = 'MSE Loss')
# pyp.xlabel('Number of Epochs')
# pyp.ylabel('Loss')
# pyp.legend()
# # pyp.show()

# pyp.figure()
# pyp.title('KLD Loss')
# pyp.plot(kld_loss, label = 'KLD Loss')
# # pyp.plot(MSE, label = 'MSE Loss')
# pyp.xlabel('Number of Epochs')
# pyp.ylabel('Loss')
# pyp.legend()
# pyp.show()

# # Plotting the Manifold
# ddt = torch.FloatTensor(dataset.ccs).to(device)
# p = torch.FloatTensor(dataset.pitches).to(device)
# v = torch.FloatTensor(dataset.velocities).to(device)
# a,b,c,z = cVAE.forward(ddt.float(),torch.cat((p.view(-1,1), v.view(-1,1)), dim=-1))
# cols = [1 if c == 'brass' else 0 for c in dataset.labels]
# pyp.figure()
# pyp.scatter(z.cpu().data.numpy()[:,0],z.cpu().data.numpy()[:,1],c = cols)
# pyp.title('Instrument Manifold')
# pyp.xlabel('LV 1')
# pyp.ylabel('LV 2')
# pyp.show()

