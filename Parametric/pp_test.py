# Script to check cc by reconstructing the audio

# Dependencies

import glob
import os
import pickle
from time import time
from scipy.signal import windows
from scipy.io.wavfile import write
from scipy import interpolate
import numpy as np
import sys
sys.path.append('../extra_dependencies/models/')
sys.path.append('../extra_dependencies/')
from hprModel import hprModelAnal,hprModelSynth
from sineModel import sineModelAnal,sineModelSynth
import morphing as m
import func_envs as fe
from essentia.standard import MonoLoader, StartStopSilence, FrameGenerator

import matplotlib.pyplot as pyp

# Insert the name of the pickle dump file here
name_dump = 'Train_octave_4_Parametric.txt'
data = pickle.load(open(name_dump, 'rb'))
# Directory to store the obtained audio files
dir_dump = './example_recon/'
try: 
    os.makedirs(dir_dump, exist_ok = True) 
    print("Directory '%s' created successfully" %dir_dump) 
except OSError as error: 
    print("Directory '%s' exists") 


# Insert the parameters here (Ensure consistency whilst generating the parametric representation i.e. the values should be same during obtaining the representation and when inverting it)
params = {}
params['fs'] = 48000
params['W'] = 1024
params['N'] = 2048
params['H'] = 256
params['t'] = -120
params['maxnSines'] = 100
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

t = 0
for k in data.keys():
	print(t + 1)
	t = t + 1

	nf = k.split('_')[0]
	f0 = dict_fmap[nf]

	new_F = np.zeros((data[k]['cc'].shape[0],params['maxnSines']))
	new_M = np.zeros_like(new_F)
	for i in range(new_F.shape[1]):
		new_F[:,i] = (i+1)*f0
	
	for j in range(new_F.shape[0]):
		fbins = np.linspace(0,params['fs'],params['N'])
		ceps_current = data[k]['cc'][j]

		zp = np.pad(ceps_current,[0 , params['N'] - len(ceps_current)],mode = 'constant',constant_values=(0, 0))
		zp = np.concatenate((zp[:params['N']//2],np.flip(zp[1:params['N']//2 + 1])))
		zp[0] = ceps_current[0]
		
		# Obtain the Envelope from the cepstrum
		specenv = np.real(np.fft.fft(zp))
		fbins = np.linspace(0,params['fs'],params['N'])
		fp = interpolate.interp1d(np.arange(params['N']),specenv,kind = 'linear',fill_value = 'extrapolate', bounds_error=False)
		new_M[j,:] = 20*fp((new_F[j,:]/params['fs'])*params['N'])


		# zp = np.pad(frame,[0,params['N'] - len(frame)],mode = 'constant',constant_values=(0, 0))
		# zp = np.concatenate((zp[:params['N']//2],np.flip(zp[1:params['N']//2 + 1])))
		# specenv = np.real(np.fft.fft(zp))
		# # print(fbins[:params['N']//2 + 1])
		# # print(specenv[:params['N']//2 + 1])
		# fp = interpolate.interp1d(np.arange(params['N']//2),specenv[:params['N']//2],kind = 'linear',fill_value = 'extrapolate', bounds_error=False)
		# new_M[j,:] = 20*fp((new_F[j,:]/params['fs'])*params['N'])

	arecon = sineModelSynth(new_F, new_M, np.empty([0,0]), params['W'], params['H'], params['fs'])
	write(filename = dir_dump + str(k) + '_recon_param.wav', rate = params['fs'], data = arecon.astype('float32'))

	
