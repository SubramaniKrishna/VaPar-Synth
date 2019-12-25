"""
Main code to generate audio from a AE by training it on a parametric model
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
from simple_ae import *
from time import time

# Setting the device to cuda if GPU is available, else to CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data into dataloader
batch_size = 512

# Here, provide the path to the pickle dump obtained as output of the parametric modeling
datafile = '../Parametric/Test_octave_4_Spectrum.txt'
# Here, you can select the MIDI pitches you want to train the network on (Should be according to the octave you choose, octave 4 - MIDI 60-71)
pitches = [60,61,70,71]

# This loads the data from the pickle dump into a dataloader
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

# Here, you can provide the hyperparameters - Latent space dimension
# It is a list as the code can be run for each specified value
latent_dims_list = [32]

# Directory to store the PyTorch .pth file (Which contains the network weights)
dir_pth_save = './AEfft/'
try: 
    os.makedirs(dir_pth_save, exist_ok = True) 
    print("Directory '%s' created successfully" %dir_pth_save) 
except OSError as error: 
    print("Directory '%s' exists") 

for ld in latent_dims_list:
	# Defining the model architecture
	dim_cc = 513
	flag_cond = True
	layer_dims_enc = [513,256,128]
	latent_dims = ld
	layer_dims_dec = [latent_dims,128,256,513]
	num_cond = 1

	now = datetime.now()
	dir_network = dir_pth_save + 'synthmanifold(' + str(now) + ')_' + str([dim_cc, flag_cond, layer_dims_enc, latent_dims, layer_dims_dec, num_cond]) + '_niters_' + str(num_epochs) + '_lr_' + str(learning_rate) + '_bsize_' + str(batch_size) +  str(pitches) +'net.pth'

	# Define the model by instantiating an object of the class
	AE = AE_synth(device = device, layer_dims_enc = layer_dims_enc, layer_dims_dec = layer_dims_dec, latent_dims = latent_dims).to(device)

	# Defining the optimizer
	optimizer = torch.optim.Adam(AE.parameters(), lr=learning_rate)

	# List to store loss
	loss_epoch = []
	MSE_loss = []
	kld_loss =[]
	test_loss = []
	# loss_avg = []
	start = time()

	for epoch in range(num_epochs):

		tmpsum = 0
		tKLD = 0
		tmptrain = 0
		it = 0
		for iteration, (label, pitch, velocity, x) in enumerate(data_loader_train):
			it = it + 1
			x, pitch, velocity = x.to(device, non_blocking=True), (pitch.float()/127).to(device, non_blocking=True), (velocity.float()/127).to(device, non_blocking=True)
			# Normalize the input
			# x = x/(abs(x).max(1, keepdim=True)[0])
			# Normalizing needed for framewise fft
			x = x/abs(x).max()
			# Ignoring the 0'th cc(only representative of volume/energy)
			# x = x[:,1:]

			x_recon,code = AE(x.float())

	        # Calculating the MSE-Loss
			calc = torch.nn.functional.mse_loss(x_recon.float(), x.float(), reduction='sum')
			loss = calc
			MSE = calc.item()

			tmptrain = tmptrain + loss.item()
			tmpsum = tmpsum + MSE
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
		# MSE.append(MSE)
		print(ld,epoch,'loss = ',loss_epoch[-1])

		# # Test Loss Computation
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


	torch.save(AE.state_dict(), dir_network)


	# # Plotting the Loss function
	# pyp.figure()
	# pyp.plot(loss_epoch,'r',label = 'training')
	# pyp.title('Training Loss')
	# pyp.xlabel('Number of Epochs')
	# pyp.ylabel('Loss')
	# pyp.legend()
	# pyp.show()

# # Plotting the Training Code Manifold
# ddt = torch.FloatTensor(dataset.ccs).to(device)
# ddt = ddt[:,1:]
# ddt/(abs(ddt).max(1, keepdim=True)[0])

# a,ztot = AE.forward(ddt)
# cols = [1 if c == 'brass' else 0 for c in dataset.labels]
# pyp.figure()
# pyp.scatter(ztot.cpu().data.numpy()[:,0],ztot.cpu().data.numpy()[:,1],c = cols)
# pyp.title('Instrument Manifold')
# pyp.xlabel('Code dimension 1')
# pyp.ylabel('Code dimension 2')
# pyp.show()