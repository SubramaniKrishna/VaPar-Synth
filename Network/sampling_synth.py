# Set of functions to sample from the latent space and synthesize the corresponding audio

# Dependencies
import numpy as np
from scipy.signal import windows
from scipy import interpolate


import sys
sys.path.append('../')
sys.path.append('../../models/')
from sineModel import sineModelAnal,sineModelSynth
from hprModel import hprModelAnal,hprModelSynth
from stft import stftAnal,stftSynth

# N-D Random walks to sample closeby points in the latent space(for sustain sound generation)
def rand_walk(start_point, step_size, num_iters, sigma = 1):
    """
    Function to initiate a random walk from a starting point
    
    Inputs
    ------
    start_point : vector(1-D Numpy array)
        Vector of the initial point of the walk
    step_size : float
        Size of random step
    num_iters : integer
        Number of random walks
    sigma : float(>0)
    	Variance of the random walk
    
    Outputs
    -------
    walk_locs : ndarray
        Matrix whose columns depict the location at each instant, and number of columns depict the number of walks
    """
    
    dim = start_point.shape[0]
    walk_locs = np.zeros((dim,num_iters))
    
    walk_locs[:,0] = start_point
    
    for i in range(1,num_iters):
        w = step_size * np.random.normal(0,sigma,dim)
        walk_locs[:,i] = walk_locs[:,i - 1] + w
    
    return walk_locs

# Use the below function sequentially after passing the output of the above through the decoder to obtain the reconstructed cepstral coeffs
def recon_samples_ls(matrix_ceps_coeffs,midi_pitch, params, f_ref = 440, choice_f = 0):
	"""
	Returns the audio corresponding to an overlap add of each of the frames reconstructed from the latent variables in walk_locs
	Note : The input should be in log dB (log|X|)
	Inputs
	------
	matrix_ceps_coeffs : np.ndarray
		Matrix whose columns depict the cepstral frames(sequential)
	midi_pitch : list of int(0 < midi_pitch < 128)
		List of MIDI number of the pitch at each time frame(can directly feed in the NSynth parameter)(same as the number of columns in the above input matrix)
		If input is a single number, that will be the pitch for all the frames
	params : dict
		Parameter dictionary for the harmonic reconstruction containing the following keys
			- fs : integer
				Sampling rate of the audio
			- W : integer
				Window size(number of frames)
			- N : integer
				FFT size(multiple of 2)
			- H : integer
				Hop size
			- nH : integer
				Number of harmonics to synthesize
	f_ref : float
		Reference frequency for MIDI(440 Hz by default)
	choice_f : 0 or 1(0 by default)
		If 0, will accept MIDI pitch and convert it to Hz
		If 1, will accept and use pitch directly in Hz
	"""

	fs = params['fs']
	W = params['W']
	N = params['N']
	H = params['H']
	nH = params['nH']
	w = windows.hann(W)

	# Defining the Frequency and Magnitude matrices
	num_frames = matrix_ceps_coeffs.shape[1]

	if(type(midi_pitch) == int):
		midi_pitch = np.zeros(num_frames) + midi_pitch

	if(choice_f == 0):
		# Convert MIDI to Hz
		hz_from_midi = f_ref*(2**((midi_pitch - 69)/12.0))
		f0 = hz_from_midi
	else:
		f0 = midi_pitch

	M = np.zeros((num_frames, nH))
	F = np.zeros((num_frames, nH))
	
	for j in range(num_frames):
		for i in range(F.shape[1]):
			F[j,i] = (i+1)*f0[j]

	# Sample the frequencies from the envelope at each instant
	for i in range(num_frames):
		# Flip and append the array to give a real frequency signal to the fft input
		ceps_current = matrix_ceps_coeffs[:,i]
		# Pad with zeros
		cc_real = np.pad(ceps_current,[0 , N - len(ceps_current)],mode = 'constant',constant_values=(0, 0))
		cc_real = np.concatenate((cc_real[:N//2],np.flip(cc_real[1:N//2 + 1])))
		cc_real[0] = ceps_current[0]
		
		# Obtain the Envelope from the cepstrum
		specenv = np.real(np.fft.fft(cc_real))
		fbins = np.linspace(0,fs,N)
		fp = interpolate.interp1d(np.arange(params['N']),specenv,kind = 'linear',fill_value = 'extrapolate', bounds_error=False)
		M[i,:] = 20*fp((F[i,:]/fs)*N)


	audio_recon = sineModelSynth(F, M, np.empty([0,0]), W, H, fs)

	return audio_recon






