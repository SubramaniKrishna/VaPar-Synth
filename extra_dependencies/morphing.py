"""
Set of functions to perform sound transformations using the Sinusoidal/Harmonic model. The main ones are - 
1) Pitch shifting with timbre preservation.
2) Timbre morphing between two sounds(with same pitch)
"""

# __________Dependencies_______________

# Include Directory for SMS-Tools here and func_envs here
import sys
sys.path.append('../')
sys.path.append('../../models/')
from sineModel import sineModelAnal,sineModelSynth
from hprModel import hprModelAnal,hprModelSynth
from stft import stftAnal,stftSynth
import func_envs as fe

from scipy.signal import windows
from scipy import interpolate
import numpy as np
import random

import essentia.standard as ess

def pitch_shifting(audio_inp, params, factor,choice,choice_recon):
	"""
	Shifts the pitch by the scalar factor given as the input.

	Depending on the choice, performs interpolation to preserve the timbre when shifting the pitch. Also returns sound with or without the original residue added.

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
	choice : 0 or 1
		If 0, simply shifts the pitch without amplitude interpolation
		If 1, performs amplitude interpolation framewise to preserve timbre
	choice_recon : 0 or 1
		If 0, returns only the sinusoidal reconstruction
		If 1, adds the original residue as well to the sinusoidal
		
	Returns
	-------
	audio_transformed : np.array
	    Returns the transformed signal in the time domain
    Residue : np.array
    	The residue of the signal
	"""

	fs = params['fs']
	W = params['W']
	N = params['N']
	H = params['H']
	t = params['t']
	maxnSines = params['maxnSines']

	w = windows.hann(W)

	F,M,P,R = hprModelAnal(x = audio_inp, fs = fs, w = w, N = N, H = H, t = t, nH = maxnSines, minSineDur = 0.02, minf0 = 10, maxf0 = 400, f0et = 5, harmDevSlope = 0.01)

	scaled_F = factor*F

	if(choice == 0):
		new_M = M
	else:
		new_M = M
		for i in range(F.shape[0]):
			# Performing the envelope interpolation framewise
			f = interpolate.interp1d(F[i,:],M[i,:],kind = 'linear',fill_value = -100, bounds_error=False)
			new_M[i,:] = f(scaled_F[i,:])

	if(choice_recon == 0):
		audio_transformed = sineModelSynth(scaled_F, new_M, np.empty([0,0]), W, H, fs)
	else:
		audio_transformed = hprModelSynth(scaled_F, new_M, np.empty([0,0]), R, W, H, fs)[0]

	return audio_transformed,R

def pitch_shift_te(audio_inp, params, factor, choice_recon, params_ceps):
	"""
	Shifts the pitch by the scalar factor given as the input.

	Performs interpolation by using the True Envelope of the Spectra. Also returns sound with or without the original residue added.

	Parameters
	----------
	audio_inp : np.array
		Numpy array containing the audio signal, in the time domain 
	params : dict
		Parameter dictionary for the sine model) containing the following keys
			- fs : integer
				Sampling rate of the audio
			- W : integer
				Window size(number of frames)
			- N : integer
				FFT size(multiple of 2)
			- H : integer
				Hop size
			- t : float
				Threshold for sinusoidal detection in dB
			- maxnSines : integer
				Number of sinusoids to detect
	factor : float
		Shift factor for the pitch. New pitch = f * (old pitch)
	choice_recon : 0 or 1
		If 0, returns only the sinusoidal reconstruction
		If 1, adds the original residue as well to the sinusoidal
	params_ceps : dict
		Parameter Dictionary for the true envelope estimation containing the following keys
			- thresh : float
				Threshold(in dB) for the true envelope estimation
			- ceps_coeffs : integer
				Number of cepstral coefficients to keep in the true envelope estimation
			- num_iters : integer
				Upper bound on number of iterations(if no convergence)
				
	Returns
	-------
	audio_transformed : np.array
	    Returns the transformed signal in the time domain
    residue : np.array
    	Residue of the original signal
	"""

	fs = params['fs']
	W = params['W']
	N = params['N']
	H = params['H']
	t = params['t']
	maxnSines = params['maxnSines']
	thresh = params_ceps['thresh']
	ceps_coeffs = params_ceps['ceps_coeffs']
	num_iters = params_ceps['num_iters']


	w = windows.hann(W)

	F,M,P,R = hprModelAnal(x = audio_inp, fs = fs, w = w, N = N, H = H, t = t, nH = maxnSines, minSineDur = 0.1, minf0 = 10, maxf0 = 1000, f0et = 5, harmDevSlope = 0.01)

	scaled_F = factor*F
	
	new_M = M
	for i in range(F.shape[0]):
		# Performing the envelope interpolation framewise(normalized log(dividing the magnitude by 20))
		f = interpolate.interp1d(F[i,:],M[i,:]/20,kind = 'linear',fill_value = -5, bounds_error=False)
		# Frequency bins
		fbins = np.linspace(0,fs/2,N)
		finp = f(fbins)
		specenv,_,_ = fe.calc_true_envelope_spectral(finp,N,thresh,ceps_coeffs,num_iters)
		# Now, once the spectral envelope is obtained, define an interpolating function based on the spectral envelope
		# fp = interpolate.interp1d(np.linspace(0,fs/2,N),np.pad(specenv[0:N//2],[0,N//2],mode = 'constant',constant_values=(0, -5)),kind = 'linear',fill_value = -5, bounds_error=False)
		fp = interpolate.interp1d(fbins[:N//2 + 1],specenv[:N//2 + 1],kind = 'linear',fill_value = 'extrapolate', bounds_error=False)
		new_M[i,:] = 20*fp(scaled_F[i,:])

	if(choice_recon == 0):
		audio_transformed = sineModelSynth(scaled_F, new_M, np.empty([0,0]), W, H, fs)
	else:
		audio_transformed = hprModelSynth(scaled_F, new_M, np.empty([0,0]), R, W, H, fs)[0]

	return audio_transformed,R

def pitch_shifting_harmonic(audio_inp, params, params_ceps, factor,choice,choice_recon,f0):
	"""
	Shifts the pitch by the scalar factor given as the input. But, assumes the sound is harmonic and hence uses only the amplitudes sampled at multiples of the fundamental frequency.
	Note : Will only perform well for harmonic/sustained sounds.
	Depending on the choice, performs interpolation to preserve the timbre when shifting the pitch. Also returns sound with or without the original residue added.

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
	maxnSines = params['maxnSines']
	thresh = params_ceps['thresh']
	ceps_coeffs = params_ceps['ceps_coeffs']
	num_iters = params_ceps['num_iters']

	w = windows.hann(W)

	F,M,P,R = hprModelAnal(x = audio_inp, fs = fs, w = w, N = N, H = H, t = t, nH = maxnSines, minSineDur = 0.02, minf0 = 10, maxf0 = 1000, f0et = 5, harmDevSlope = 0.01)
	
	new_F= np.zeros_like(F)
	for i in range(F.shape[1]):
		new_F[:,i] = (i+1)*f0

	scaled_F = factor*new_F

	if(choice == 0):
		new_M = M
	elif(choice == 1):
		new_M = M
		for i in range(F.shape[0]):
			# Performing the envelope interpolation framewise
			f = interpolate.interp1d(F[i,:],M[i,:],kind = 'linear',fill_value = -100, bounds_error=False)
			new_M[i,:] = f(scaled_F[i,:])
	else:
		new_M = M
		for i in range(F.shape[0]):
			# Performing the envelope interpolation framewise(normalized log(dividing the magnitude by 20))
			f = interpolate.interp1d(F[i,:],M[i,:]/20,kind = 'linear',fill_value = -5, bounds_error=False)
			# Frequency bins
			fbins = np.linspace(0,fs/2,2*N)
			finp = f(fbins)
			specenv,_,_ = fe.calc_true_envelope_spectral(finp,N,thresh,ceps_coeffs,num_iters)
			# Now, once the spectral envelope is obtained, define an interpolating function based on the spectral envelope
			fp = interpolate.interp1d(fbins[:N//2 + 1],specenv[:N//2 + 1],kind = 'linear',fill_value = 'extrapolate', bounds_error=False)
			new_M[i,:] = 20*fp(scaled_F[i,:])

	if(choice_recon == 0):
		audio_transformed = sineModelSynth(scaled_F, new_M, np.empty([0,0]), W, H, fs)
	else:
		audio_transformed = hprModelSynth(scaled_F, new_M, np.empty([0,0]), R, W, H, fs)[0]

	return audio_transformed

def residue_lpc(audio_inp, params,lpc_order):
	"""
	Obtains the LPC representation of the Residual Spectral(LPC envelope), and then generates the residual by IFFT'ing this representation with random phase.

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
	lpc_order : integer
		Number of coefficients in the LPC representation
		
	Returns
	-------
	res_transformed : np.array
	    Returns the transformed residue(LPC envelope approximation) in the time domain
	"""

	fs = params['fs']
	W = params['W']
	N = params['N']
	H = params['H']
	t = params['t']
	maxnSines = params['maxnSines']

	w = windows.hann(W)

	F,M,P,R = hprModelAnal(x = audio_inp, fs = fs, w = w, N = N, H = H, t = t, nH = maxnSines, minSineDur = 0.02, minf0 = 10, maxf0 = 400, f0et = 5, harmDevSlope = 0.01)
	harmonics_recon = sineModelSynth(tfreq = F, tmag = M, tphase = P, N = W, H = H, fs = fs)

	# Initializing an empty list to store the residual spectral approximations(LPC)
	xmX = []

	# Normalize the Residue before analysis(throws a np zero error otherwise)
	nf = np.max(np.abs(R))
	# nf = 1
	# print(nf)

	R = R/nf
	
	for frame in ess.FrameGenerator(R.astype('float32'), W, H):
		inp = np.pad(frame,[0,N - W],mode = 'constant',constant_values=(0, 0))
		env_frame = fe.lpc_envelope(inp,lpc_order,fs,len(inp)//2 + 1)
		xmX.append(env_frame)
	xmX = np.array(xmX)
	XpX = 2*np.pi*np.random.rand(xmX.shape[0],xmX.shape[1])

	# xmX,XpX = stftAnal(audio_inp,w,N,H)
	# Obtain the audio from the above representation
	res_transformed =  stftSynth(xmX, XpX, W, H)*nf

	# ***Re-normalize the Residual so that it lies in the same range as the original residue***
	# scale_init = np.max(np.abs(audio_inp))/np.max(np.abs(R))
	# scale_final = np.max(np.abs(harmonics_recon))/scale_init
	res_transformed = (res_transformed/np.max(np.abs(res_transformed)))


	return res_transformed

def SF_transform_pitch_res(audio_inp, params, factor, params_ceps,lpc_order):
	"""
	Wrapper function to shift the pitch by factor, and to transform the Residue using the SF model and add it back after the shift.

	Parameters
	----------
	audio_inp : np.array
		Numpy array containing the audio signal, in the time domain 
	params : dict
		Parameter dictionary for the sine model) containing the following keys
			- fs : integer
				Sampling rate of the audio
			- W : integer
				Window size(number of frames)
			- N : integer
				FFT size(multiple of 2)
			- H : integer
				Hop size
			- t : float
				Threshold for sinusoidal detection in dB
			- maxnSines : integer
				Number of sinusoids to detect
	factor : float
		Shift factor for the pitch. New pitch = f * (old pitch)
	params_ceps : dict
		Parameter Dictionary for the true envelope estimation containing the following keys
			- thresh : float
				Threshold(in dB) for the true envelope estimation
			- ceps_coeffs : integer
				Number of cepstral coefficients to keep in the true envelope estimation
			- num_iters : integer
				Upper bound on number of iterations(if no convergence)
	lpc_order : integer
		Number of coefficients in the LPC representation
				
	Returns
	-------
	audio_transformed : np.array
	    Returns the transformed audio signal in the time domain
	"""

	harmonics_shifted,R = pitch_shift_te(audio_inp, params, factor, 0, params_ceps)
	residue_sf = residue_lpc(audio_inp, params,lpc_order)

	harmonics_shifted = harmonics_shifted/np.max(np.abs(harmonics_shifted))
	scale_init = (np.max(np.abs(R))/np.max(np.abs(audio_inp)))
	residue_sf = residue_sf * scale_init
	
	return harmonics_shifted[:min(harmonics_shifted.size,residue_sf.size)] + residue_sf[:min(harmonics_shifted.size,residue_sf.size)]
	

def morph_samepitch_te(audio_inp1, audio_inp2, alpha, f0, params, params_ceps):
	"""
	Timbre morphing between two sounds of same pitch by linearly interpolating the true envelope.

	Parameters
	----------
	audio_inp1 : np.array
		Numpy array containing the first audio signal, in the time domain
	audio_inp2 : np.array
		Numpy array containing the second audio signal, in the time domain 
	alpha : float
		Interpolation factor(0 <= alpha <= 1), alpha*audio1 + (1 - alpha)*audio2
	f0 : float
		Fundamental Frequency(to reconstruct harmonics)
	params : dict
		Parameter dictionary for the sine model) containing the following keys
			- fs : integer
				Sampling rate of the audio
			- W : integer
				Window size(number of frames)
			- N : integer
				FFT size(multiple of 2)
			- H : integer
				Hop size
			- t : float
				Threshold for sinusoidal detection in dB
			- maxnSines : integer
				Number of sinusoids to detect
	params_ceps : dict
		Parameter Dictionary for the true envelope estimation containing the following keys
			- thresh : float
				Threshold(in dB) for the true envelope estimation
			- ceps_coeffs : integer
				Number of cepstral coefficients to keep in the true envelope estimation
			- num_iters : integer
				Upper bound on number of iterations(if no convergence)
				
	Returns
	-------
	audio_morphed : np.array
		Returns the morphed audio in the time domain
	"""

	fs = params['fs']
	W = params['W']
	N = params['N']
	H = params['H']
	t = params['t']
	maxnSines = params['maxnSines']
	thresh = params_ceps['thresh']
	ceps_coeffs = params_ceps['ceps_coeffs']
	num_iters = params_ceps['num_iters']

	w = windows.hann(W)

	F1,M1,_,_ = hprModelAnal(x = audio_inp1, fs = fs, w = w, N = N, H = H, t = t, nH = maxnSines, minSineDur = 0.02, minf0 = 10, maxf0 = 400, f0et = 5, harmDevSlope = 0.01)
	F2,M2,_,_ = hprModelAnal(x = audio_inp2, fs = fs, w = w, N = N, H = H, t = t, nH = maxnSines, minSineDur = 0.02, minf0 = 10, maxf0 = 400, f0et = 5, harmDevSlope = 0.01)

	# Defining the frequency matrix as multiples of the harmonics
	new_F= np.zeros_like(F1 if F1.shape[0] < F2.shape[0] else F2)
	for i in range(new_F.shape[1]):
		new_F[:,i] = (i+1)*f0

	# Defining the Magnitude matrix
	new_M = np.zeros_like(M1 if M1.shape[0] < M2.shape[0] else M2)

	for i in range(new_M.shape[0]):
		# Performing the envelope interpolation framewise(normalized log(dividing the magnitude by 20))
		f1 = interpolate.interp1d(F1[i,:],M1[i,:]/20,kind = 'linear',fill_value = -100, bounds_error=False)
		f2 = interpolate.interp1d(F2[i,:],M2[i,:]/20,kind = 'linear',fill_value = -100, bounds_error=False)
		# Frequency bins
		fbins = np.linspace(0,fs/2,N)
		finp1 = f1(fbins)
		finp2 = f2(fbins)
		specenv1,_,_ = fe.calc_true_envelope_spectral(finp1,N,thresh,ceps_coeffs,num_iters)
		specenv2,_,_ = fe.calc_true_envelope_spectral(finp2,N,thresh,ceps_coeffs,num_iters)
		# Interpolate based on the true envelopes
		specenv = alpha*specenv1 + (1 - alpha)*specenv2

		# fp = interpolate.interp1d(np.linspace(0,fs/2,N),np.pad(specenv[0:N//2],[0,N//2],mode = 'constant',constant_values=(0, -5)),kind = 'linear',fill_value = 'extrapolate', bounds_error=False)
		fp = interpolate.interp1d(fbins[:N//2 + 1],specenv[:N//2 + 1],kind = 'linear',fill_value = -10, bounds_error=False)
		new_M[i,:] = 20*fp(new_F[i,:])

	audio_morphed = sineModelSynth(new_F, new_M, np.empty([0,0]), W, H, fs)

	return audio_morphed


def morph_samepitch_cc(audio_inp1, audio_inp2, alpha, f0, params, params_ceps):
	"""
	Timbre morphing between two sounds of same pitch by linearly interpolating the cepstral representation of the true envelope.

	Parameters
	----------
	audio_inp1 : np.array
		Numpy array containing the first audio signal, in the time domain
	audio_inp2 : np.array
		Numpy array containing the second audio signal, in the time domain 
	alpha : float
		Interpolation factor(0 <= alpha <= 1), alpha*audio1 + (1 - alpha)*audio2
	f0 : float
		Fundamental Frequency(to reconstruct harmonics)
	params : dict
		Parameter dictionary for the sine model) containing the following keys
			- fs : integer
				Sampling rate of the audio
			- W : integer
				Window size(number of frames)
			- N : integer
				FFT size(multiple of 2)
			- H : integer
				Hop size
			- t : float
				Threshold for sinusoidal detection in dB
			- maxnSines : integer
				Number of sinusoids to detect
	params_ceps : dict
		Parameter Dictionary for the true envelope estimation containing the following keys
			- thresh : float
				Threshold(in dB) for the true envelope estimation
			- ceps_coeffs : integer
				Number of cepstral coefficients to keep in the true envelope estimation
			- num_iters : integer
				Upper bound on number of iterations(if no convergence)
				
	Returns
	-------
	audio_morphed : np.array
		Returns the morphed audio in the time domain
	"""

	fs = params['fs']
	W = params['W']
	N = params['N']
	H = params['H']
	t = params['t']
	maxnSines = params['maxnSines']
	thresh = params_ceps['thresh']
	ceps_coeffs = params_ceps['ceps_coeffs']
	num_iters = params_ceps['num_iters']

	w = windows.hann(W)

	F1,M1,_,_ = hprModelAnal(x = audio_inp1, fs = fs, w = w, N = N, H = H, t = t, nH = maxnSines, minSineDur = 0.02, minf0 = 10, maxf0 = 400, f0et = 5, harmDevSlope = 0.01)
	F2,M2,_,_ = hprModelAnal(x = audio_inp2, fs = fs, w = w, N = N, H = H, t = t, nH = maxnSines, minSineDur = 0.02, minf0 = 10, maxf0 = 400, f0et = 5, harmDevSlope = 0.01)

	# Defining the frequency matrix as multiples of the harmonics
	new_F= np.zeros_like(F1 if F1.shape[0] < F2.shape[0] else F2)
	for i in range(new_F.shape[1]):
		new_F[:,i] = (i+1)*f0

	# Defining the Magnitude matrix
	new_M = np.zeros_like(M1 if M1.shape[0] < M2.shape[0] else M2)

	for i in range(new_M.shape[0]):
		# Performing the envelope interpolation framewise(normalized log(dividing the magnitude by 20))
		f1 = interpolate.interp1d(F1[i,:],M1[i,:]/20,kind = 'linear',fill_value = -100, bounds_error=False)
		f2 = interpolate.interp1d(F2[i,:],M2[i,:]/20,kind = 'linear',fill_value = -100, bounds_error=False)
		# Frequency bins
		fbins = np.linspace(0,fs/2,N)
		finp1 = f1(fbins)
		finp2 = f2(fbins)
		specenv1,_,_ = fe.calc_true_envelope_spectral(finp1,N,thresh,ceps_coeffs,num_iters)
		specenv2,_,_ = fe.calc_true_envelope_spectral(finp2,N,thresh,ceps_coeffs,num_iters)

		# Obtain the Cepstral Representation of the True envelopes
		cc_te_1 = np.real(np.fft.ifft(specenv1))
		cc_te_2 = np.real(np.fft.ifft(specenv2))

		# Linearly interpolate the cepstral coefficients, and reconstruct the true envelope from that
		cc_interp = alpha*cc_te_1 + (1 - alpha)*cc_te_2
		specenv = np.real(np.fft.fft(cc_interp))

		# fp = interpolate.interp1d(np.linspace(0,fs/2,N),np.pad(specenv[0:N//2],[0,N//2],mode = 'constant',constant_values=(0, -5)),kind = 'linear',fill_value = 'extrapolate', bounds_error=False)
		fp = interpolate.interp1d(fbins[:N//2 + 1],specenv[:N//2 + 1],kind = 'linear',fill_value = -10, bounds_error=False)
		new_M[i,:] = 20*fp(new_F[i,:])

	audio_morphed = sineModelSynth(new_F, new_M, np.empty([0,0]), W, H, fs)

	return audio_morphed

def morph_samepitch_lpc(audio_inp1, audio_inp2, alpha, f0, params, params_ceps):
	"""
	Timbre morphing between two sounds of same pitch by linearly interpolating the lpc representation of the true envelope(obtained from its cepstral representation).

	Parameters
	----------
	audio_inp1 : np.array
		Numpy array containing the first audio signal, in the time domain
	audio_inp2 : np.array
		Numpy array containing the second audio signal, in the time domain 
	alpha : float
		Interpolation factor(0 <= alpha <= 1), alpha*audio1 + (1 - alpha)*audio2
	f0 : float
		Fundamental Frequency(to reconstruct harmonics)
	params : dict
		Parameter dictionary for the sine model) containing the following keys
			- fs : integer
				Sampling rate of the audio
			- W : integer
				Window size(number of frames)
			- N : integer
				FFT size(multiple of 2)
			- H : integer
				Hop size
			- t : float
				Threshold for sinusoidal detection in dB
			- maxnSines : integer
				Number of sinusoids to detect
	params_ceps : dict
		Parameter Dictionary for the true envelope estimation containing the following keys
			- thresh : float
				Threshold(in dB) for the true envelope estimation
			- ceps_coeffs : integer
				Number of cepstral coefficients to keep in the true envelope estimation
			- num_iters : integer
				Upper bound on number of iterations(if no convergence)
				
	Returns
	-------
	audio_morphed : np.array
		Returns the morphed audio in the time domain
	"""

	fs = params['fs']
	W = params['W']
	N = params['N']
	H = params['H']
	t = params['t']
	maxnSines = params['maxnSines']
	thresh = params_ceps['thresh']
	ceps_coeffs = params_ceps['ceps_coeffs']
	num_iters = params_ceps['num_iters']

	w = windows.hann(W)

	F1,M1,_,_ = hprModelAnal(x = audio_inp1, fs = fs, w = w, N = N, H = H, t = t, nH = maxnSines, minSineDur = 0.02, minf0 = 10, maxf0 = 400, f0et = 5, harmDevSlope = 0.01)
	F2,M2,_,_ = hprModelAnal(x = audio_inp2, fs = fs, w = w, N = N, H = H, t = t, nH = maxnSines, minSineDur = 0.02, minf0 = 10, maxf0 = 400, f0et = 5, harmDevSlope = 0.01)

	# Defining the frequency matrix as multiples of the harmonics
	new_F= np.zeros_like(F1 if F1.shape[0] < F2.shape[0] else F2)
	for i in range(new_F.shape[1]):
		new_F[:,i] = (i+1)*f0

	# Defining the Magnitude matrix
	new_M = np.zeros_like(M1 if M1.shape[0] < M2.shape[0] else M2)

	for i in range(new_M.shape[0]):
		# print('frame ',i,' of ',new_M.shape[0])
		# Performing the envelope interpolation framewise(normalized log(dividing the magnitude by 20))
		f1 = interpolate.interp1d(F1[i,:],M1[i,:]/20,kind = 'linear',fill_value = -100, bounds_error=False)
		f2 = interpolate.interp1d(F2[i,:],M2[i,:]/20,kind = 'linear',fill_value = -100, bounds_error=False)
		# Frequency bins
		fbins = np.linspace(0,fs/2,N)
		finp1 = f1(fbins)
		finp2 = f2(fbins)
		specenv1,_,_ = fe.calc_true_envelope_spectral(finp1,N,thresh,ceps_coeffs,num_iters)
		specenv2,_,_ = fe.calc_true_envelope_spectral(finp2,N,thresh,ceps_coeffs,num_iters)

		# Obtain the Cepstral Representation of the True envelopes
		cc_te_1 = np.real(np.fft.ifft(specenv1))
		cc_te_2 = np.real(np.fft.ifft(specenv2))

		# Obtaining the LPC Representation from the Cepstral Representation
		lpc_cc_te_1 = fe.cc_to_lpc(cc_te_1,ceps_coeffs)
		lpc_cc_te_2 = fe.cc_to_lpc(cc_te_2,ceps_coeffs)

		# Interpolate the lpc's, and reconvert back to cepstral coefficients
		lpc_interp = alpha*lpc_cc_te_1 + (1 - alpha)*lpc_cc_te_2
		cc_interp = fe.lpc_to_cc(lpc_interp,ceps_coeffs + 1 ,ceps_coeffs)
		# Pad with zeros(Done to reduce number of computations)
		cc_interp = np.pad(cc_interp,[0 , N - len(cc_interp)],mode = 'constant',constant_values=(0, 0))

		# Flip and append the array to give a real frequency signal to the fft input
		cc_interp = np.concatenate((cc_interp[:N//2],np.flip(cc_interp[1:N//2 + 1])))

		# Interpolating the Zeroth coefficient separately(it represents the gain/power of the signals)
		cc_interp[0] = alpha*cc_te_1[0] + (1 - alpha)*cc_te_2[0]

		specenv = np.real(np.fft.fft(cc_interp))

		# fp = interpolate.interp1d(np.linspace(0,fs/2,N),np.pad(specenv[0:N//2],[0,N//2],mode = 'constant',constant_values=(0, -5)),kind = 'linear',fill_value = 'extrapolate', bounds_error=False)
		fp = interpolate.interp1d(fbins[:N//2 + 1],specenv[:N//2 + 1],kind = 'linear',fill_value = -10, bounds_error=False)
		new_M[i,:] = 20*fp(new_F[i,:])

	audio_morphed = sineModelSynth(new_F, new_M, np.empty([0,0]), W, H, fs)

	return audio_morphed

def morph_samepitch_lsf(audio_inp1, audio_inp2, alpha, f0, params, params_ceps):
	"""
	Timbre morphing between two sounds of same pitch by linearly interpolating the lsf representation of the true envelope(obtained from its lpc,cepstral representation).

	Parameters
	----------
	audio_inp1 : np.array
		Numpy array containing the first audio signal, in the time domain
	audio_inp2 : np.array
		Numpy array containing the second audio signal, in the time domain 
	alpha : float
		Interpolation factor(0 <= alpha <= 1), alpha*audio1 + (1 - alpha)*audio2
	f0 : float
		Fundamental Frequency(to reconstruct harmonics)
	params : dict
		Parameter dictionary for the sine model) containing the following keys
			- fs : integer
				Sampling rate of the audio
			- W : integer
				Window size(number of frames)
			- N : integer
				FFT size(multiple of 2)
			- H : integer
				Hop size
			- t : float
				Threshold for sinusoidal detection in dB
			- maxnSines : integer
				Number of sinusoids to detect
	params_ceps : dict
		Parameter Dictionary for the true envelope estimation containing the following keys
			- thresh : float
				Threshold(in dB) for the true envelope estimation
			- ceps_coeffs : integer
				Number of cepstral coefficients to keep in the true envelope estimation
			- num_iters : integer
				Upper bound on number of iterations(if no convergence)
				
	Returns
	-------
	audio_morphed : np.array
		Returns the morphed audio in the time domain
	"""

	fs = params['fs']
	W = params['W']
	N = params['N']
	H = params['H']
	t = params['t']
	maxnSines = params['maxnSines']
	thresh = params_ceps['thresh']
	ceps_coeffs = params_ceps['ceps_coeffs']
	num_iters = params_ceps['num_iters']

	w = windows.hann(W)

	F1,M1,_,_ = hprModelAnal(x = audio_inp1, fs = fs, w = w, N = N, H = H, t = t, nH = maxnSines, minSineDur = 0.02, minf0 = 10, maxf0 = 400, f0et = 5, harmDevSlope = 0.01)
	F2,M2,_,_ = hprModelAnal(x = audio_inp2, fs = fs, w = w, N = N, H = H, t = t, nH = maxnSines, minSineDur = 0.02, minf0 = 10, maxf0 = 400, f0et = 5, harmDevSlope = 0.01)

	# Defining the frequency matrix as multiples of the harmonics
	new_F= np.zeros_like(F1 if F1.shape[0] < F2.shape[0] else F2)
	for i in range(new_F.shape[1]):
		new_F[:,i] = (i+1)*f0

	# Defining the Magnitude matrix
	new_M = np.zeros_like(M1 if M1.shape[0] < M2.shape[0] else M2)

	for i in range(new_M.shape[0]):
		# print('frame ',i,' of ',new_M.shape[0])
		# Performing the envelope interpolation framewise(normalized log(dividing the magnitude by 20))
		f1 = interpolate.interp1d(F1[i,:],M1[i,:]/20,kind = 'linear',fill_value = -100, bounds_error=False)
		f2 = interpolate.interp1d(F2[i,:],M2[i,:]/20,kind = 'linear',fill_value = -100, bounds_error=False)
		# Frequency bins
		fbins = np.linspace(0,fs/2,N)
		finp1 = f1(fbins)
		finp2 = f2(fbins)
		specenv1,_,_ = fe.calc_true_envelope_spectral(finp1,N,thresh,ceps_coeffs,num_iters)
		specenv2,_,_ = fe.calc_true_envelope_spectral(finp2,N,thresh,ceps_coeffs,num_iters)

		# Obtain the Cepstral Representation of the True envelopes
		cc_te_1 = np.real(np.fft.ifft(specenv1))
		cc_te_2 = np.real(np.fft.ifft(specenv2))

		# Define number of LPC(LSF) coefficients to keep
		# Cannot keep all, as precision error causes the coefficients to blow up
		L = 60
		# Obtaining the LPC Representation from the Cepstral Representation
		lpc_cc_te_1 = fe.cc_to_lpc(cc_te_1,L)
		lpc_cc_te_2 = fe.cc_to_lpc(cc_te_2,L)

		# Obtain LSF representation from the LPC
		lsf_lpc_cc_te_1 = fe.lpc_to_lsf(lpc_cc_te_1)
		lsf_lpc_cc_te_2 = fe.lpc_to_lsf(lpc_cc_te_2)

		# Interpolate the LSF and convert LSF back to LPC
		lsf_interp = alpha*lsf_lpc_cc_te_1 + (1 - alpha)*lsf_lpc_cc_te_2
		lpc_interp = fe.lsf_to_lpc(lsf_interp)

		# Reconvert LPC's to CC's
		cc_interp = fe.lpc_to_cc(lpc_interp,L + 1 ,L)
		# Pad with zeros(Done to reduce number of computations)
		cc_interp = np.pad(cc_interp,[0 , N - len(cc_interp)],mode = 'constant',constant_values=(0, 0))

		# Flip and append the array to give a real frequency signal to the fft input
		cc_interp = np.concatenate((cc_interp[:N//2],np.flip(cc_interp[1:N//2 + 1])))

		# Interpolating the Zeroth coefficient separately(it represents the gain/power of the signals)
		cc_interp[0] = alpha*cc_te_1[0] + (1 - alpha)*cc_te_2[0]

		specenv = np.real(np.fft.fft(cc_interp))

		# fp = interpolate.interp1d(np.linspace(0,fs/2,N),np.pad(specenv[0:N//2],[0,N//2],mode = 'constant',constant_values=(0, -5)),kind = 'linear',fill_value = 'extrapolate', bounds_error=False)
		fp = interpolate.interp1d(fbins[:N//2 + 1],specenv[:N//2 + 1],kind = 'linear',fill_value = -10, bounds_error=False)
		new_M[i,:] = 20*fp(new_F[i,:])

	audio_morphed = sineModelSynth(new_F, new_M, np.empty([0,0]), W, H, fs)

	return audio_morphed


def sustain_sound_gen(audio_inp, params, params_ceps, f0, rwl, alpha):
	"""
	Re-synthesizes the input audio using a random walk starting from the middle frame of the audio.

	Parameters
	----------
	audio_inp : np.array
		Numpy array containing the audio signal, in the time domain 
	params : dict
		Parameter dictionary for the sine model) containing the following keys
			- fs : integer
				Sampling rate of the audio
			- W : integer
				Window size(number of frames)
			- N : integer
				FFT size(multiple of 2)
			- H : integer
				Hop size
			- t : float
				Threshold for sinusoidal detection in dB
			- maxnSines : integer
				Number of sinusoids to detect
	params_ceps : dict
		Parameter Dictionary for the true envelope estimation containing the following keys
			- thresh : float
				Threshold(in dB) for the true envelope estimation
			- ceps_coeffs : integer
				Number of cepstral coefficients to keep in the true envelope estimation
			- num_iters : integer
				Upper bound on number of iterations(if no convergence)
	f0 : float
		Fundamental frequency(or pitch) of the note
	rwl : Integer
		Number of hops to consider around the middle frame
	alpha : float(0<alpha<1)
		Closeness to the current frame(for continuity of the spectral frames during reconstruction)
				
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
	maxnSines = params['maxnSines']
	thresh = params_ceps['thresh']
	ceps_coeffs = params_ceps['ceps_coeffs']
	num_iters = params_ceps['num_iters']


	w = windows.hann(W)

	F,M,P,R = hprModelAnal(x = audio_inp, fs = fs, w = w, N = N, H = H, t = t, nH = maxnSines, minSineDur = 0.02, minf0 = 10, maxf0 = 1000, f0et = 5, harmDevSlope = 0.01)

	new_F= np.zeros_like(F)
	for i in range(F.shape[1]):
		new_F[:,i] = (i+1)*f0
	new_M = M
	
	# Initial parameters for random walk
	midpoint = F.shape[0]//2 # Selecting the middle frame 
	current_frame = midpoint
	f = interpolate.interp1d(F[current_frame,:],M[current_frame,:]/20,kind = 'linear',fill_value = -5, bounds_error=False)
	# Frequency bins
	fbins = np.linspace(0,fs/2,N)
	finp = f(fbins)
	specenv_at,_,_ = fe.calc_true_envelope_spectral(finp,N,thresh,ceps_coeffs,num_iters)

	# Reconstruct the Magnitude array from the frequency array(only the middle frame but)
	for i in range(M.shape[0]):
		
		# Updating the current frame as per a random walk update(add upper and lower threshold)
		current_frame = current_frame + random.choice([-rwl,rwl])
		if(current_frame >= M.shape[0] - 1):
			current_frame = M.shape[0] - 1
		if(current_frame <= 0):
			current_frame = 0
		f = interpolate.interp1d(F[current_frame,:],M[current_frame,:]/20,kind = 'linear',fill_value = -5, bounds_error=False)
		# Frequency bins
		fbins = np.linspace(0,fs/2,N)
		finp = f(fbins)
		specenv_new,_,_ = fe.calc_true_envelope_spectral(finp,N,thresh,ceps_coeffs,num_iters)
		# Pnce the initial and final envelopes are obtained, interpolate to obtain the new(intermediate) envelope
		# The closer the envelope is to 1, the less the envelope will change from its current value
		
		specenv_at = alpha*specenv_at + (1 - alpha)*specenv_new
		# Now, once the spectral envelope is obtained, define an interpolating function based on the spectral envelope
		fp = interpolate.interp1d(fbins[:N//2 + 1],specenv_at[:N//2 + 1],kind = 'linear',fill_value = 'extrapolate', bounds_error=False)
		new_M[i,:] = 20*fp(new_F[i,:])

	# Reconstruction of the sound ignoring the residual
	audio_transformed = sineModelSynth(new_F, new_M, np.empty([0,0]), W, H, fs)

	return audio_transformed