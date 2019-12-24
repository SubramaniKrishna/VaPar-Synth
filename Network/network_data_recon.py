import torch
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as pyp
from vae_krishna import *
import sys
from dataset import *
import glob
import ast
sys.path.append('../extra_dependencies/models/')
from hprModel import hprModelAnal,hprModelSynth
from sineModel import sineModelAnal,sineModelSynth
from essentia.standard import MonoLoader
from scipy.io.wavfile import write
from scipy.signal import windows
from scipy import interpolate
import pickle
import os
# Importing our model
from simple_ae import *
import librosa
import sampling_synth as ss

# Load the data i.e. path to the Pickle dump file obtained as output of the parametric representation
# Here, you can provide either the Train or Test representation
# Note, when evaluating the network, make sure to evaluate on Test (or data that the network has not been trained on)
datafile = '../Parametric/Train_octave_4_Parametric.txt'
data = pickle.load(open(datafile, 'rb'))


# -----------------------------------------------CVAE----------------------------------------------------------
# Provide directory where CVAE network outputs are stored
# Load the parameters from the strig of the .pth file
dir_files = './CVAEparams/'
list_pth_files = glob.glob(dir_files + '*.pth')

# Display the available parameter files
print('Available state files to load are(for cVAE) : ')
for it,i in enumerate(list_pth_files):
	print(it,i.split('/')[-1][:-4])

# Choose the .pth file
idx = (int)(input('Choose the input index'))
# print(list_pth_files[idx])
# Load the parameters into a list
list_params = ast.literal_eval(list_pth_files[idx].split('_')[1])
file_load_VAE = list_pth_files[idx]


# list params, make the model and load the wights from the .pth file
# Fix device here(currently cpu)
device = 'cpu'
# device = 'cuda'
# Defining the model architecture
dim_cc = list_params[0]
flag_cond = list_params[1]
layer_dims_enc = list_params[2]
latent_dims = list_params[3]
layer_dims_dec = list_params[4]
num_cond = list_params[5]

cVAE = cVAE_synth(flag_cond = flag_cond, layer_dims_enc = layer_dims_enc, layer_dims_dec = layer_dims_dec, latent_dims = latent_dims, num_cond = num_cond, device = device)
cVAE.load_state_dict(torch.load(file_load_VAE,map_location = 'cpu'))


# -----------------------------------------------AE----------------------------------------------------------
# Provide directory where AE network outputs are stored
# Load the parameters from the strig of the .pth file
dir_files = './AEparams/'
list_pth_files = glob.glob(dir_files + '*.pth')

# Display the available parameter files
print('Available state files to load are(for AE) : ')
for it,i in enumerate(list_pth_files):
	print(it,i.split('/')[-1][:-4])

# Choose the .pth file
idx = (int)(input('Choose the input index'))
# print(list_pth_files[idx])
# Load the parameters into a list
list_params = ast.literal_eval(list_pth_files[idx].split('_')[1])
file_load_AE = list_pth_files[idx]


# list params, make the model and load the wights from the .pth file
# Fix device here(currently cpu)
device = 'cpu'
# device = 'cuda'
# Defining the model architecture
dim_cc = list_params[0]
flag_cond = list_params[1]
layer_dims_enc = list_params[2]
latent_dims = list_params[3]
layer_dims_dec = list_params[4]
num_cond = list_params[5]

AE = AE_synth(device = device, layer_dims_enc = layer_dims_enc, layer_dims_dec = layer_dims_dec, latent_dims = latent_dims).to(device)
AE.load_state_dict(torch.load(file_load_AE,map_location = 'cpu'))


# Provide the parameters for the reconstruction here (ensure consistency i.e. use the same parameters that were used to obtain the parameteric representation)
params = {}
params['fs'] = 48000
params['W'] = 1024
params['N'] = 2048
params['H'] = 256
params['t'] = -120
params['maxnSines'] = 100
params['nH'] = 100
params_ceps = {}
params_ceps['thresh'] = 0.1
params_ceps['num_iters'] = 10000
octave = 4

# Dictionary to map frequencies
dict_fmap = {}
start_frequency = octave*55
step = 2**(1.0/12)
dict_fmap['C'] = (step**(3))*start_frequency
dict_fmap['C#'] = step*dict_fmap['C']
dict_fmap['D'] = step*dict_fmap['C#']
dict_fmap['D#'] = step*dict_fmap['D']
dict_fmap['E'] = step*dict_fmap['D#']
dict_fmap['F'] = step*dict_fmap['E']
dict_fmap['F#'] = step*dict_fmap['F']
dict_fmap['G'] = step*dict_fmap['F#']
dict_fmap['G#'] = step*dict_fmap['G']
dict_fmap['A'] = step*dict_fmap['G#']
dict_fmap['A#'] = step*dict_fmap['A']
dict_fmap['B'] = step*dict_fmap['A#']

# Here, you can select the pitches you want to evaluate
# Ensure that the selected pitches are present in your generated pickle dump file
pl = [60,61,62,63,64,65,66,67,68,69,70,71]

error_dict = {}
error_dict['cVAE'] = {k:[0,0] for k in pl}
error_dict['VAE'] = {k:[0,0] for k in pl}
error_dict['AE'] = {k:[0,0] for k in pl}

# plot_folder = './plot_folder/'

# Provide the directory to store the network reconstructed audio
dir_recon_audio = './dir_recon_audio/'
try: 
    os.makedirs(dir_recon_audio, exist_ok = True) 
    print("Directory '%s' created successfully" %dir_recon_audio) 
except OSError as error: 
    print("Directory '%s' exists") 


t = 0
for k in data.keys():
	nf = k.split('_')[0]
	f0 = dict_fmap[nf]
	midival = (int)(69 + 12*np.log2(f0/440))

	if ((midival in pl) == False):
		continue

	print(t + 1)
	t = t + 1

	# new_F = np.zeros((data[k]['cc'].shape[0],params['maxnSines']))
	# new_M = np.zeros_like(new_F)
	# for i in range(new_F.shape[1]):
	# 	new_F[:,i] = (i+1)*f0
	# new_M_og = np.zeros_like(new_M)
	
	# # Obtain the cc's by forward passing the data through the net
	cc_in = data[k]['cc']
	ddt = torch.FloatTensor(cc_in)
	# # ddt = ddt/(abs(ddt).max(1, keepdim=True)[0])
	# # ddt = ddt/abs(ddt).max()
	cc_norm = ddt.data.numpy()
	# # ddt = ddt[:,1:]
	p = torch.FloatTensor(midival*np.ones(cc_in.shape[0]))
	x_recon_cVAE,mu,sig,ztot = cVAE.forward(ddt,(p.float()/127).view(-1,1))
	# ddt_inp = torch.cat((ddt.float(),p.view(-1,1).float()/127),dim = -1)
	x_recon_AE,code_AE = AE.forward(ddt)
	# x_recon_AE = x_recon_AE[:,:-1]

	# Error Computation analysis per pitch
	# print(ddt.shape)
	e_cVAE = torch.nn.functional.mse_loss(ddt.float(), x_recon_cVAE.float(), reduction='sum').item()
	e_AE = torch.nn.functional.mse_loss(ddt.float(), x_recon_AE.float(), reduction='sum').item()
	error_dict['cVAE'][midival][0] = error_dict['cVAE'][midival][0] + e_cVAE
	error_dict['cVAE'][midival][1] = error_dict['cVAE'][midival][1] + ddt.shape[0]
	error_dict['AE'][midival][0] = error_dict['AE'][midival][0] + e_AE
	error_dict['AE'][midival][1] = error_dict['AE'][midival][1] + ddt.shape[0]

	# # cc_recon = np.vstack((-1*np.ones((1,cc_in.shape[0])),x_recon.data.numpy().T))
	cc_recon_cVAE = x_recon_cVAE.data.numpy().T
	cc_recon_AE = x_recon_AE.data.numpy().T
	# # print(cc_in.shape,cc_recon.shape)
	a_og = ss.recon_samples_ls(matrix_ceps_coeffs = cc_in.T, midi_pitch = midival, params = params)
	a_recon_cVAE = ss.recon_samples_ls(matrix_ceps_coeffs = cc_recon_cVAE, midi_pitch = midival, params = params)
	a_recon_AE = ss.recon_samples_ls(matrix_ceps_coeffs = cc_recon_AE, midi_pitch = midival, params = params)

	# # for j in range(new_F.shape[0]):
	# # 	fbins = np.linspace(0,params['fs']/2,params['N'])
	# # 	frame = cc_recon.T[j]
	# # 	zp = np.pad(frame,[0,params['N'] - len(frame)],mode = 'constant',constant_values=(0, 0))
	# # 	zp = np.concatenate((zp[:params['N']//2],np.flip(zp[1:params['N']//2 + 1])))
	# # 	specenv = np.real(np.fft.fft(zp))
	# # 	fp = interpolate.interp1d(fbins[:params['N']//2 + 1],specenv[:params['N']//2 + 1],kind = 'linear',fill_value = '-5', bounds_error=False)
	# # 	new_M[j,:] = 20*fp(new_F[j,:])

	# # 	frame = cc_norm[j]
	# # 	zp = np.pad(frame,[0,params['N'] - len(frame)],mode = 'constant',constant_values=(0, 0))
	# # 	zp = np.concatenate((zp[:params['N']//2],np.flip(zp[1:params['N']//2 + 1])))
	# # 	specenv = np.real(np.fft.fft(zp))
	# # 	fp = interpolate.interp1d(fbins[:params['N']//2 + 1],specenv[:params['N']//2 + 1],kind = 'linear',fill_value = 'extrapolate', bounds_error=False)
	# # 	new_M_og[j,:] = 20*fp(new_F[j,:])

	# # a_recon = sineModelSynth(new_F, new_M, np.empty([0,0]), params['W'], params['H'], params['fs'])
	# # a_og = sineModelSynth(new_F, new_M_og, np.empty([0,0]), params['W'], params['H'], params['fs'])
	write(filename = dir_recon_audio + str(k) + '_og.wav', rate = params['fs'], data = a_og.astype('float32'))
	write(filename = dir_recon_audio + str(k) + '_recon_AE.wav', rate = params['fs'], data = a_recon_AE.astype('float32'))
	write(filename = dir_recon_audio + str(k) + '_recon_cVAE.wav', rate = params['fs'], data = a_recon_cVAE.astype('float32'))

	# # Plots
	# # Average +- 5 envelope plots from the centre
	# # num_f = 5
	# # tf = 2*num_f + 1
	# # te_avg_og = 0
	# # te_avg_cVAE = 0
	# # te_avg_AE = 0
	# # fbins = np.arange(params['N'])*(params['fs']/(params['N']))
	# # f_ind = (int)(0.41*params['N'])
	# # for j in range(len(cc_norm)//2 - num_f,len(cc_norm)//2 + num_f):
	# # 	te_avg_og = te_avg_og + np.real(np.fft.fft(cc_norm[j],params['N']))[:f_ind]
	# # 	te_avg_cVAE = te_avg_cVAE + np.real(np.fft.fft(cc_recon_cVAE.T[j],params['N']))[:f_ind]
	# # 	te_avg_AE = te_avg_AE + np.real(np.fft.fft(cc_recon_AE.T[j],params['N']))[:f_ind]

	# # te_avg_og = te_avg_og/tf
	# # te_avg_cVAE = te_avg_cVAE/tf
	# # te_avg_AE = te_avg_AE/tf

	# # te_avg_og[0] = 0
	# # te_avg_cVAE[0] = 0
	# # te_avg_AE[0] = 0

	# # pyp.figure()
	# # pyp.title('MIDI : ' + str(midival))
	
	# # num_ind = 0
	# # cc_norm[len(cc_norm)//2][0] = 0
	# # cc_recon_cVAE.T[len(cc_recon_cVAE)//2][0] = 0
	# # cc_recon_AE.T[len(cc_recon_AE)//2][0] = 0
	# # pyp.plot(fbins[:f_ind],np.real(np.fft.fft(cc_norm[len(cc_norm)//2],params['N']))[:f_ind],label = 'original')
	# # pyp.plot(fbins[:f_ind],np.real(np.fft.fft(cc_recon_cVAE.T[len(cc_recon_cVAE)//2],params['N']))[:f_ind],label = 'cVAE')
	# # pyp.plot(fbins[:f_ind],np.real(np.fft.fft(cc_recon_AE.T[len(cc_recon_AE)//2],params['N']))[:f_ind],label = 'AE')
	# pyp.plot(fbins[:f_ind],te_avg_og,'k-',label = 'original')
	# pyp.plot(fbins[:f_ind],te_avg_cVAE,'y-',label = 'cVAE')
	# pyp.plot(fbins[:f_ind],te_avg_AE,'c-',label = 'AE')
	# # pyp.plot(cc_norm[len(cc_norm)//2],'r.',label = 'original')
	# # pyp.plot(cc_recon_AE.T[len(cc_recon)//2],'g.',label = 'AE reconstructed')
	# pyp.xlabel('Frequency(Hz)')
	# pyp.ylabel('Magnitude(20dB)')
	# pyp.legend(loc = 'best')
	# pyp.savefig('./test/' + k + '.png')
	# pyp.close()


# Plot Error per pitch for all 3 methods
tot_err_cVAE = []
tot_err_AE = []
for p in pl:
	tot_err_cVAE.append(error_dict['cVAE'][p][0]/error_dict['cVAE'][p][1])
	tot_err_AE.append(error_dict['AE'][p][0]/error_dict['AE'][p][1])

# print(tot_err_AE)
# print(tot_err_cVAE)

pyp.figure()
pyp.title('PitchWise MSE for the 3 Algos')
pyp.plot(pl,tot_err_cVAE,label = 'cVAE')
pyp.plot(pl,tot_err_AE,label = 'AE')
pyp.xlabel('MIDI pitch')
pyp.ylabel('MSE')
pyp.legend(loc = 'best')
# pyp.savefig('./test/AllPitchwiseErrors.png')
# pyp.close()
pyp.show()
