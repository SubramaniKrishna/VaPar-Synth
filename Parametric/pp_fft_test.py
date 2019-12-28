# Script to check fft by reconstructing the audio using griffin lim

# Dependencies

import glob
import pickle
from time import time
import os
from scipy.signal import windows
from scipy.io.wavfile import write
from scipy import interpolate
import numpy as np
import sys
sys.path.append('../extra_dependencies/models/')
sys.path.append('../extra_dependencies/')
from hprModel import hprModelAnal,hprModelSynth
from sineModel import sineModelAnal,sineModelSynth
from stft import stftSynth
import morphing as m
import func_envs as fe
from essentia.standard import MonoLoader, StartStopSilence, FrameGenerator
import matplotlib.pyplot as pyp
import librosa

# Insert the name of the pickle dump file here
name_dump = 'Test_octave_4_Spectrum.txt'
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
params['N'] = 1024
params['H'] = 256
params['t'] = -120
params['maxnSines'] = 50
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

	stft_inp = data[k]['cc']
	# print(stft_inp[:,:params['N']//2 + 1].shape)
	# phase_inp = data[k]['phase']
	# Using griffin lim to invert
	# arecon = reconstruct_signal_griffin_lim(magnitude_spectrogram = 10**(stft_inp[:,:params['N']//2 + 1]), fft_size = params['N'], hopsamp = params['H'], iterations = 100)
	arecon = librosa.core.griffinlim(S = 10**(stft_inp.T),hop_length = params['H'])

	# If you are using the magnitude CQT instead of the magnitude FFT, uncomment the following line (instead of the above) to invert the CQT
	# Take care to use the same set of parameters for inversion as you used when obtaining the CQT
	# arecon = librosa.griffinlim_cqt(stft_inp, sr=params['fs'],hop_length = params['H'], bins_per_octave=36)

	# arecon = stftSynth(20*stft_inp, np.pi*np.random.rand(phase_inp.shape[0],phase_inp.shape[1]), params['W'] , params['H'] )

	write(filename = dir_dump + str(k) + '_recon_fft.wav', rate = params['fs'], data = arecon/np.max(abs(arecon)).astype('float32'))