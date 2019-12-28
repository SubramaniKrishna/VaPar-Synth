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
import sampling_synth as ss
from scipy.signal import windows
from scipy import interpolate
import pickle
from scipy.signal import stft
import os
# Importing our model
from simple_ae import *
import librosa
import csv

# Load the data i.e. path to the Pickle dump file obtained as output of the parametric representation
# Here, you can provide either the Train or Test representation
# Note, when evaluating the network, make sure to evaluate on Test (or data that the network has not been trained on)
datafile = '../Parametric/Test_octave_4_Spectrum.txt'
data = pickle.load(open(datafile, 'rb'))

# -----------------------------------------------CVAE----------------------------------------------------------
# Provide directory where CVAE network outputs are stored
# Load the parameters from the strig of the .pth file
dir_files = './CVAEfft/'
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
dir_files = './AEfft/'
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
octave = 4
params = {}
params['H'] = 256
params['N'] = 1024
params['fs'] = 48000

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
error_dict['AE'] = {k:[0,0] for k in pl}


# plot_folder = './plot_folder/'

# Provide the directory to store the network reconstructed audio
dir_recon_audio = './dir_recon_audio_fft/'
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

	# Load STFT matrix from data 
	stft_in = data[k]['cc']
	# Normalizing factor
	nf =  np.max(abs(stft_in))
	stft_norm = torch.FloatTensor(stft_in/nf)

	p = torch.FloatTensor(midival*np.ones(stft_in.shape[0]))
	x_recon_cVAE,mu,sig,ztot = cVAE.forward(stft_norm,(p.float()/127).view(-1,1))
	x_recon_AE,code_AE = AE.forward(stft_norm)
	stft_recon_cVAE = x_recon_cVAE.data.numpy().T
	stft_recon_AE = x_recon_AE.data.numpy().T

	e_cVAE = torch.nn.functional.mse_loss(stft_norm.float(), x_recon_cVAE.float(), reduction='sum').item()/stft_in.shape[0]
	e_AE = torch.nn.functional.mse_loss(stft_norm.float(), x_recon_AE.float(), reduction='sum').item()/stft_in.shape[0]
	error_dict['cVAE'][midival][0] = error_dict['cVAE'][midival][0] + e_cVAE
	error_dict['cVAE'][midival][1] = error_dict['cVAE'][midival][1] + 1
	error_dict['AE'][midival][0] = error_dict['AE'][midival][0] + e_AE
	error_dict['AE'][midival][1] = error_dict['AE'][midival][1] + 1

	# Denormalize and invert the log
	stft_recon_cVAE = 10**(stft_recon_cVAE*nf)
	stft_recon_AE = 10**(stft_recon_AE*nf)

	# Griffin Lim to invert specgram to obtain the audio
	a_og = librosa.core.griffinlim(S = 10**(stft_in.T),hop_length = params['H'])
	a_recon_stft_cVAE = librosa.core.griffinlim(S = stft_recon_cVAE ,hop_length = params['H'])
	a_recon_stft_AE = librosa.core.griffinlim(S = stft_recon_AE,hop_length = params['H'])

	# If you are using the CQT representation, uncomment the following lines to invert the audio
	# Take care to use the same parameters while inversion that you used to obtain the CQT
	"""
	stft_recon_cVAE = np.log10(stft_recon_cVAE)
	stft_recon_AE = np.log10(stft_recon_AE)
	a_og = librosa.griffinlim_cqt(stft_in.T, sr=params['fs'],hop_length = params['H'], bins_per_octave=36)
	a_recon_stft_cVAE = librosa.griffinlim_cqt(stft_recon_cVAE, sr=params['fs'],hop_length = params['H'], bins_per_octave=36)
	a_recon_stft_AE = librosa.griffinlim_cqt(stft_recon_AE, sr=params['fs'],hop_length = params['H'], bins_per_octave=36)
	"""


	# STFT plots
	# f_og, t_og, Zxx_og = stft(a_og, params['fs'], nperseg=params['N'])
	# f_recon, t_recon, Zxx_recon = stft(a_recon_stft_AE, params['fs'], nperseg=params['N'])
	# print(f_recon.shape,Zxx_recon.shape)

	# Save data to plotting format -> each row : (x,y,f(x,y))
	# with open('./test_fft/' + str(k) + '_og.csv', 'w') as writeFile:
	# 	writer = csv.writer(writeFile)
	# 	writer.writerow(['x','y','data'])
	# 	for y in range(f_og[:107].shape[0]):
	# 		for x in range(50):
	# 			print([y,x,20*np.log10(np.abs(Zxx_og[y,x]))])
	# 			writer.writerow([t_og[x],f_og[y],(20*np.log10(np.abs(Zxx_og[y,x])))])
	# 		writer.writerow('')
	# 	writeFile.close()
	# with open('./test_fft/' + str(k) + '_recon.csv', 'w') as writeFile:
	# 	writer = csv.writer(writeFile)
	# 	writer.writerow(['x','y','data'])
	# 	for y in range(f_recon[:107].shape[0]):
	# 		for x in range(50):
	# 			print([y,x,20*np.log10(np.abs(Zxx_og[y,x]))])
	# 			writer.writerow([t_recon[x],f_recon[y],(20*np.log10(np.abs(Zxx_recon[y,x])))])
	# 		writer.writerow('')
	# 	writeFile.close()


	# pyp.figure()
	# pyp.title('Original')
	# # pyp.subplot(211)
	# # pyp.title('Original')
	# pyp.pcolormesh(t_og, f_og[:107], 20*np.log10(np.abs(Zxx_og[:107,:])))
	# pyp.ylabel('Frequency [Hz]')
	# pyp.xlabel('Time [sec]')
	# pyp.colorbar()
	# pyp.savefig('./test_fft/' + str(k) + '_og.png')
	# # pyp.subplot(212)
	# pyp.close()
	# pyp.figure()
	# pyp.title('Reconstructed')
	# pyp.pcolormesh(t_recon, f_recon[:107], 20*np.log10(np.abs(Zxx_recon[:107,:])))
	# pyp.ylabel('Frequency [Hz]')
	# pyp.xlabel('Time [sec]')
	# # pyp.tight_layout()
	# pyp.colorbar()
	# pyp.savefig('./test_fft/' + str(k) + '_skipped_recon.png')
	# # pyp.show()

	write(filename = dir_recon_audio + str(k) + '_ogGL.wav', rate = params['fs'], data = a_og/np.max(abs(a_og)).astype('float32'))
	write(filename = dir_recon_audio + str(k) + '_recon_stft_CVAE.wav', rate = params['fs'], data = a_recon_stft_cVAE/np.max(abs(a_recon_stft_AE)).astype('float32'))
	write(filename = dir_recon_audio + str(k) + '_recon_stft_AE.wav', rate = params['fs'], data = a_recon_stft_AE/np.max(abs(a_recon_stft_AE)).astype('float32'))

	# Plots
	# pyp.figure()
	# pyp.title(nf + '_' + str(midival))
	# Pxx, freqs, bins, im = pyp.plot.specgram(a_og/np.max(abs(a_og)), NFFT=params['N'], Fs=params['fs'], noverlap=256)
	# plot.xlabel('Time')
	# plot.ylabel('Frequency')
	# pyp.show()
	# pyp.savefig('./test/' + k + '_og.png')
	# fbins = np.arange(params['N'])*(params['fs']/(params['N']))
	# pyp.plot(fbins[:params['N']//2],np.real(np.fft.fft(cc_norm[0],params['N']))[:params['N']//2],label = 'original')
	# pyp.plot(fbins[:params['N']//2],np.real(np.fft.fft(cc_recon_cVAE.T[0],params['N']))[:params['N']//2],label = 'cVAE')
	# pyp.plot(fbins[:params['N']//2],np.real(np.fft.fft(cc_recon_AE.T[0],params['N']))[:params['N']//2],label = 'AE')
	# # pyp.plot(cc_norm[len(cc_norm)//2],'r.',label = 'original')
	# # pyp.plot(cc_recon_AE.T[len(cc_recon)//2],'g.',label = 'AE reconstructed')
	# pyp.xlabel('Frequency(Hz)')
	# pyp.ylabel('Magnitude(20dB)')
	# pyp.legend(loc = 'best')
	# pyp.savefig('./test/' + k + '.png')
	# pyp.close()

	# stft_inp = data[k]['cc']
	# sn = torch.FloatTensor(stft_inp/np.max(np.abs(stft_inp)))
	# stft_recon ,code = AE(sn.float())

	# a_og = librosa.core.griffinlim(S = 10**(stft_inp.T),hop_length = params['H'])
	# a_recon_stft = librosa.core.griffinlim(S = 10**(stft_recon.data.numpy().T*np.max(np.abs(stft_inp))),hop_length = params['H'])

	# write(filename = './test/' + str(k) + '_og.wav', rate = params['fs'], data = a_og/np.max(abs(a_og)).astype('float32'))
	# write(filename = './test/' + str(k) + '_recon_stft.wav', rate = params['fs'], data = a_recon_stft/np.max(abs(a_recon_stft)).astype('float32'))

# Plot Error per pitch for all 3 methods
tot_err_cVAE = []
tot_err_AE = []
for p in pl:
	tot_err_cVAE.append(error_dict['cVAE'][p][0]/error_dict['cVAE'][p][1])
	tot_err_AE.append(error_dict['AE'][p][0]/error_dict['AE'][p][1])

pyp.figure()
pyp.title('PitchWise MSE for the 3 Algos')
pyp.plot(pl,tot_err_cVAE,label = 'cVAE')
pyp.plot(pl,tot_err_AE,label = 'AE')
pyp.xlabel('MIDI pitch')
pyp.ylabel('MSE')
pyp.legend(loc = 'best')
# pyp.savefig('./test_fft/AllPitchwiseErrors.png')
# pyp.close()
pyp.show()