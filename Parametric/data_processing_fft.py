"""
Keep in mind
	1. Normalization(invert whatever normalization you perform in pre-processing during synthesis)
"""


# Dependencies

import glob
import pickle
from time import time
from scipy.signal import windows
from scipy import interpolate
import numpy as np
import sys
sys.path.append('../extra_dependencies/models/')
sys.path.append('../extra_dependencies/')
from hprModel import hprModelAnal,hprModelSynth
from sineModel import sineModelAnal,sineModelSynth
from stft import stftAnal
import morphing as m
import func_envs as fe
from essentia.standard import MonoLoader, StartStopSilence, FrameGenerator
import librosa


def spec_calc(audio_inp, params):
	"""
	Calculates the framewise cepstral coefficients for the true envelope of the audio file.

	Parameters
	----------
	audio_inp : np.array
		Numpy array containing the audio signal, in the time domain 
	params : dict
		Parameter dictionary for the sine model) containing the following keys
			- fs : Sampling rate of the audio
			- W : Window size(number of frames)
			- N : FFT size(multiple of 2)
			- H : Hop size
			- t : Threshold for sinusoidal detection in dB
			- maxnSines : Number of sinusoids to detect
	factor : float
		Shift factor for the pitch. New pitch = f * (old pitch)
	choice : 0,1,2
		If 0, simply shifts the pitch without amplitude interpolation
		If 1, performs amplitude interpolation framewise to preserve timbre
		If 2, uses the True envelope of the amplitude spectrum to sample the points from
	choice_recon : 0 or 1
		If 0, returns only the sinusoidal reconstruction
		If 1, adds the original residue as well to the sinusoidal
	f0 : Hz
		The fundamental frequency of the note
		
	Returns
	-------
	audio_transformed : np.array
	    Returns the transformed signal in the time domain
	"""

	fs = params['fs']
	W = params['W']
	N = params['N']
	H = params['H']
	t = params['t']

	w = windows.hann(W)

	# Compute the STFT
	xmX, xpX = stftAnal(x = audio_inp, w = w, N = N, H = H)
	# xmX = stft_for_reconstruction(x = audio_inp, fft_size = N, hopsamp = H)
	# Remove the dB normalization done in the above function
	xmX = xmX/20

	return xmX,xpX


###############################################################################################

# Defining the parameters and parameters dicts (All these can be changed accordingly)
params = {}
params['fs'] = 48000
params['W'] = 1024
params['N'] = 1024
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

# Start______________________________________________
# Point this to the Train/Test directory defined before via the data-loading code
choice_traintest = 'Test'

dir_audio = '../Data_Loading/sounds_folder/' + str(choice_traintest) + '/'
name_dump = choice_traintest + '_octave_' + str(octave) + '_Spectrum'

list_files_to_process = glob.glob(dir_audio + '*.wav')

results = {}
p = 0

print("\nPreprocessing started...\n")
N = len(list_files_to_process)
start = time()
for it,f in enumerate(list_files_to_process):
	# if(it == 200):
		# break
	progress = (it*100/N)
	if progress > p:
		current = time()
		p+=1
		print('\033[F\033[A\033[K[' + '='*int((progress//5)) + ' '*int((20-(progress//5))) + '] ', int(progress), '%', '\t Time elapsed: ',current-start, ' seconds')

	ainp = MonoLoader(filename = f, sampleRate = params['fs'])()
	fHz = dict_fmap[f.split('/')[-1].split('_')[0]]

	# Extract relevant information(name,midi)
	name = f.split('/')[-1][:-4]
	midival = (int)(69 + 12*np.log2(fHz/440))

	# Select the relevant portion of audio to compute cc using essentia's silence detection function
	s_3 = StartStopSilence(threshold = -30)
	for frame in FrameGenerator(ainp, frameSize = params['N'], hopSize = params['H']):
		sss = s_3(frame)
	start_frame = (int)(sss[0]*params['H'])
	stop_frame = (int)(sss[1]*params['H'])
	ainp = ainp[start_frame:stop_frame]

	# Compute the magnitude specgram
	op, ph = spec_calc(audio_inp = ainp, params = params)

	# If you want to use the magnitude CQT as the input representation insted of the FFT, replace the above line by uncommenting the following lines
	"""
	# This assumes 8 octaves at 36 bins per octave, can be changed (also change this accordingly during the inversion stage!)
	C = librosa.cqt(y=ainp, sr=params['fs'],hop_length = params['H'], bins_per_octave=36, n_bins = 8*36)
	op = np.abs(C)
	ph = np.angle(C)
	"""


	# Store cc'c + other relevant parameters in dict
	results[name] = {}
	results[name]['pitch'] = midival
	results[name]['velocity'] = 50
	results[name]['cc'] = op[:,:params['N']//2 + 1]
	results[name]['phase'] = ph

# Write the inputs to a pickle dump
pickle.dump(results, open(name_dump + '.txt', 'wb'))