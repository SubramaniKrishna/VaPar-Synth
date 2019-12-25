"""
Code to sample from the latent space and generate the corresponding audio. The input is the conditional parameter pitch.
"""

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

# Provide the parameters for the Generation/Synthesis here 
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

# Provide the directory to store the network generated audio
dir_gen_audio = './dir_netgen_audio/'
try: 
    os.makedirs(dir_gen_audio, exist_ok = True) 
    print("Directory '%s' created successfully" %dir_gen_audio) 
except OSError as error: 
    print("Directory '%s' exists") 

# Dimensionality of latent space
ld = latent_dims
# Specify the duration of the vibrato note in seconds
dur_n = 3
nf = (int)(params['fs']/(params['H'])*dur_n)
"""
You can take the pitch inputs in two ways (setting the choice variable to '0' or '1')
choice = 0:
	Here, you can manually input the pitch contour. Just specify the start and end frequencies. A matplotlib plot will pop, asking you to click at points.
	Each point you click is a (pitch,time) pair, and the more points you add, the finer the sampling. The pitch contour will be formed by interpolating appropriately
	Once you have specified the contour, close the matplotlib popup window.
choice = 1:
	Here, a single note with vibrato is generated. You can specify the vibrato parameters as needed.
An audio file will be saved in the specified directory
"""
choice = 0

if(choice == 0):
	# ______________________________________________________________________________________________________________________________________
	# Choice = 0;
	# Obtaining the Pitch Contour by drawing on matplotlib
	# Obtaining the Contour by passing (pitch,time) coordinates and linearly interpolating the frequencies in between

	# Starting Frequency (Specify)
	f_start = 250
	# Ending frequency
	f_end = 500


	lp_x = []
	lp_y = []

	class LineBuilder:
	    def __init__(self, line):
	        self.line = line
	        self.xs = list(line.get_xdata())
	        self.ys = list(line.get_ydata())
	        self.cid = line.figure.canvas.mpl_connect('button_press_event', self)

	    def __call__(self, event):
	        print('click', event)
	        if event.inaxes!=self.line.axes: return
	        lp_x.append(event.xdata)
	        lp_y.append(event.ydata)
	        # print(list_points_clicked)
	        self.xs.append(event.xdata)
	        self.ys.append(event.ydata)
	        self.line.set_data(self.xs, self.ys)
	        self.line.figure.canvas.draw()

	fig = pyp.figure()
	ax = fig.add_subplot(111)
	ax.set_title('click to select the pitch points (they will be linearly interpolated')
	pyp.ylim(f_start,f_end)
	pyp.xlim(0,dur_n)
	line, = ax.plot([0], [f_start])  # empty line
	linebuilder = LineBuilder(line)
	pyp.show()



	# Specify array containing the time instants and pitches
	# The pitch contour will be formed by linearly interpolating
	# array_time_instants = np.array([0.5,1.1,2.3,2.5,2.8])
	# array_frequencies = np.array([260,290,250,350,400])
	array_time_instants = np.array(lp_x)
	array_frequencies = np.array(lp_y)
	num_points = array_frequencies.shape[0] 

	# Append the start and end frequencies to the main frequency array. Do same with time(start -> 0 and stop-> duration specified)
	array_frequencies = np.insert(array_frequencies,[0,num_points],[f_start,f_end])
	array_time_instants = np.insert(array_time_instants,[0,num_points],[0,dur_n])
	# print(array_frequencies)
	# print(array_time_instants)
	#Assuming that spacing between all frequencies is uniform (i.e. more the frequencies specified, more dense the sampling)
	# nbf = (int)(nf/num_points)
	fcontour_Hz = np.zeros(nf)

	for i in range(0,len(array_frequencies) - 1):
		s = array_time_instants[i]
		e = array_time_instants[i+1]
		# print(s,e)
		s = (int)((s/dur_n)*nf)
		e = (int)((e/dur_n)*nf)
		nbf = (e - s)
		# print(s,e)
		fr = np.linspace(array_frequencies[i],array_frequencies[i+1],nbf)
		fcontour_Hz[s:e] = fr
	# print(fcontour_Hz)

else:
	# ____________________________________________________________________________________________________________________________________
	# Choice = 1;
	# Generating a note with Vibrato (Frequency Modulation)
	# Vibrato pitch contour in Hz
	# Center Frequency in MIDI
	p = 69
	# Obtain f_c by converting the pitch from MIDI to Hz
	f_Hz = 440*2**((p-69)/12)

	# Vibrato depth(1-2% of f_c)
	Av = 0.04*f_Hz

	# Vibrato frequency(generally 5-10 Hz)
	fV_act = 6
	# Sub/sampling the frequency according to the Hop Size
	f_v = 2*np.pi*((fV_act*params['H'])/(params['fs']))

	# Forming the contour
	# The note will begin with a sustain pitch, and then transition into a vibrato
	# Specify the fraction of time the note will remain in sustain
	frac_sus = 0.25
	fcontour_Hz =np.concatenate((f_Hz*np.ones((int)(nf*frac_sus) + 1),f_Hz + Av*np.sin(np.arange((int)((1-frac_sus)*nf))*f_v)))


# Once the pitch contour is obtained, generate the sound by sampling and providing the contour as a conditional parameter.

# Convert from Hz to MIDI frequency
pch = (69 + 12*np.log2(fcontour_Hz/440))

# Obtain a trajectory in the latent space using a random walk
z_ss = 0.001*ss.rand_walk(np.zeros(ld), 0.00001, nf)
z_ss1 = torch.FloatTensor(z_ss.T)
cond_inp = torch.FloatTensor(pch)
cond_inp = cond_inp.float()/127
# print(z_ss1.shape,cond_inp.shape)
# Sample from the CVAE latent space
s_z_X = cVAE.sample_latent_space(z_ss1,cond_inp.view(-1,1))
cc_network = s_z_X.data.numpy().squeeze()

# Obtain the audio by sampling the spectral envelope at the specified pitch values
a_gen_cVAE = ss.recon_samples_ls(matrix_ceps_coeffs = cc_network.T, midi_pitch = fcontour_Hz, params = params,choice_f = 1)
write(filename = dir_gen_audio + 'gensynth_cVAE.wav', rate = params['fs'], data = a_gen_cVAE.astype('float32'))