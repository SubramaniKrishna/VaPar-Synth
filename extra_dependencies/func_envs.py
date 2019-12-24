"""Companion functions to compute various kinds of envelope functions
"""

# Dependencies
import numpy as np
from scipy.signal import windows
import scipy.linalg as sla
from scipy.signal import freqz,lfilter

import essentia.standard as ess


def real_cepstrum(signal_inp,fft_size):
	"""
	Returns Real Cepstrum of input(real) signal.

	Computes the real cepstrum as defined by the following formula :math:`c[m] = F^{-1}\{log_{10}F\{x[n]\}\}`
	Where F is the Fourier Transform and x[n] is the input signal.

	Parameters
	----------
	signal_inp : np.array
	    numpy array containing the audio signal
	fft_size : integer(even)
		FFT Size

	Returns
	-------
	cepstral_coeffs : np.array
		Returns the cepstral coefficients

	References
	----------
	.. [1] Wikipedia, "Cepstrum".
			http://en.wikipedia.org/wiki/Cepstrum

	"""

	log_sig_fft_mag = np.log10(np.abs(np.fft.fft(signal_inp,fft_size)) + 10**(-10))
	cepstral_coeffs = np.real(np.fft.ifft(log_sig_fft_mag,fft_size))

	return cepstral_coeffs


def ceps_envelope(signal_inp,fft_size,window,fs,f0,num_coeff,choice,choice_inp):
	"""
	Returns the Spectral Envelope based on the Windowed Cepstral 'Liftering' method

	Lifters the cepstrum and computes it's FFT to find the spectral envelope.

	Parameters
	----------
	signal_inp : np.array
	    numpy array containing the audio signal
     	look at choice_inp below
	fft_size : integer(even)
		FFT Size
	window : string
		Window function
	fs : integer
		Sampling rate
	f0 : integer
		Fundamental Frequency
	num_coeff : integer
		Number of cepstral coefficients to consider(0 <= num_coeff <= fft_size)
	choice : 0 or 1
		if 0, will use paper defined number of cepstral coefficients
		if 1, will use user specified number of cepstral coefficients
	choice_inp : 0 or 1
		if 0, signal_inp should be the time domain signal
		if 1, signal_inp should be the frequency domain signal(fft of the time domain signal)

	Returns
	-------
	spectral_envelope : np.array
	    Returns the spectral envelope

	References
    ----------
    .. [1] Cross Synthesis Using Cepstral Smoothing or Linear Prediction for Spectral Envelopes, J.O. Smith
           https://ccrma.stanford.edu/~jos/SpecEnv/LPC_Envelope_Example_Speech.html

	"""

	if(choice_inp == 0):
		cepstral_coeffs = real_cepstrum(signal_inp,fft_size);
	else:
		log_sig_fft_mag = np.log10(np.abs(signal_inp + 10**(-10)))
		cepstral_coeffs = np.real(np.fft.ifft(log_sig_fft_mag,fft_size))
	# Number of cepstral coefficients to keep(as defined in the True Envelope paper)
	num_paper = (int)(fs/(2*f0))
	if(choice == 0):
		R = num_paper
	else:
		R = num_coeff

	# Generate the window of appropriate size(same as the number of cepstral coefficients to keep)
	if(window == 'hann'):
		win = windows.boxcar(2*R)

	win_fin = np.zeros(fft_size)
	win_fin[0:R] = win[R:]
	win_fin[fft_size-R:] = win[:R]

	# Lifter the cepstrum
	liftered_ceps = cepstral_coeffs * win_fin
	# liftered_ceps[0] = 0

	# Finding the envelope by taking the FFT of the liftered signal
	spec_env = np.real(np.fft.fft(liftered_ceps,fft_size))

	# zero meaning
	# spec_env = spec_env - np.mean(spec_env)

	return spec_env,win_fin,liftered_ceps


def lpc(signal_inp,M):
	"""
	Returns LPC coefficients of the signal

	Computes the LPC coefficients for the given signal using the normal equations(Yule Walker system)

	Parameters
	----------
	signal_inp : np.array
	    numpy array containing the audio signal
	M : integer
	    LPC coefficients order

	Returns
	-------
	lpc_coeffs : np.array
	    Returns the cepstral coefficients

	References
    ----------
    .. [1] Wikipedia, "Linear Prediction".
           https://en.wikipedia.org/wiki/Linear_prediction

	"""

	# Computing the autocorrelation vector
	cc = (np.correlate(signal_inp,signal_inp,mode = 'full'))
	rx = cc[len(cc)//2 + 1:]

	# Forming the Toeplitz autocovariance matrix from the above vector
	R = sla.toeplitz(rx)

	# Solving the Yule-Walker system
	lpc_coeffs = -np.linalg.inv(R[0:M-1,0:M-1])*rx[1:M]

	return lpc_coeffs


def lpc_envelope(signal_inp,M,fs,freq_size):
	"""
	Returns the Spectral Envelope based on the LPC method

	Finds the spectral envelope by finding the frequency response of an IIR filter with coefficients as the lp coefficients

	Parameters
	----------
	signal_inp : np.array
	    numpy array containing the audio signal
	M : integer
	    LPC coefficients order
    fs : float
    	Sampling Rate
	freq_size : integer
		Size of the output frequency envelope

	Returns
	-------
	spectral_envelope : np.array
	    Returns the spectral envelope

	References
	----------
	.. [1] Cross Synthesis Using Cepstral Smoothing or Linear Prediction for Spectral Envelopes, J.O. Smith
	       https://ccrma.stanf2000ord.edu/~jos/SpecEnv/LPC_Envelope_Example_Speech.html

	"""	
	# Find the lpc coefficients using the above function
	# lpc_coeffs = lpc(signal_inp,M)
	lpc_coeffs = ess.LPC(order = M,sampleRate = fs)(signal_inp)
	# print(lpc_coeffs[0])

	# To obtain the normalization constant for the filter
	res_e = lfilter(b = lpc_coeffs[0],a = 1,x = signal_inp)
	G = np.linalg.norm(res_e)
	# print(G)

	# Frequency response of the IIR filter with the above as it's denominator coefficients 
	w, h = freqz(b = G,a = lpc_coeffs[0],worN = freq_size,whole = True)

	# log transform the above
	spectral_envelope = 20*np.log10(np.abs(h)[0:freq_size//2 + 1])

	#zero mean
	# spectral_envelope = spectral_envelope - np.mean(spectral_envelope)

	return spectral_envelope


def calc_true_envelope(signal_inp,fft_size,thresh,num_coeff,stopping_iters):
	"""
	Returns the Spectral Envelope based on the iterative version of the Windowed Cepstral 'Liftering' method

	Iteratively pushes the windowed liftered cepstral envelope towards the 'true' envelope

	Parameters
	----------
	signal_inp : np.array
	    numpy array containing the audio signal in the time domain
	fft_size : integer(even)
	    FFT Size
	thresh : float
		The stopping criteria for the final envelope(Stop when final lies within init +- thresh), dB value
	num_coeff : integer
		Number of coefficients to consider for the cepstrum
	stopping_iters : integer
		Upper bound on number of iterations(if no convergence)


	Returns
	-------
	spectral_envelope : np.array
	    Returns the spectral envelope computed by the true method
	cou : int
		Number of iterations required to converge
	env_list : list(np.arrays)
		List containing the spectral envelope for each iteration 

	References
	----------
	.. [1] Röbel, Axel, and Xavier Rodet. "Efficient spectral envelope estimation and its application to pitch shifting and envelope preservation." International Conference on Digital Audio Effects. 2005.

	"""

	A_ip1 = np.log10(np.abs(np.fft.fft(signal_inp,fft_size)))
	A_0 = A_ip1
	env_list = []

	# Threshold array
	thresh_arr = thresh*np.ones(fft_size)
	
	cou = 0
	while(True):

		# Adaptive Cepstral Update to speedup
		# Here, c_im1 <-> C_i in the paper abd c_p <-> C_i' in the paper.
		V_i,w,c = ceps_envelope(10**(A_ip1),fft_size,'hann',44100,100,num_coeff,1,1)
		# c_im1 = c
		# c_im1 = np.real(np.fft.ifft(V_i))
		A_ip1 = np.where((A_ip1 > V_i),A_ip1,V_i)
		# c_p = np.real(np.fft.ifft(A_ip1))
		# print(np.max(c_im1),np.min(c_im1),np.max(c_p),np.min(c_p))
		# Computing the In-Band and Out-of-Band Energies
		# E_i = np.linalg.norm((c_p - c_im1)[:num_coeff])**2
		# E_o = np.linalg.norm((c_p - c_im1)[num_coeff + 1:fft_size//2 + 1])**2
		# Computing the Adaptive weighting factor
		# adaptive_lambda = ((E_i + E_o)/E_i)
		# adaptive_lambda = 1
		# c_p = adaptive_lambda*(c_p - c_im1) + c_im1
		# A_ip1 = np.real(np.fft.fft(c_p))

		# print('iteration : ',cou + 1)
		cou = cou + 1

		env_list.append(A_ip1)

		# Stopping Criteria
		if((((A_0 - V_i) <= thresh_arr).all()) or (cou >= stopping_iters)):
			Vf = V_i
			break

	return Vf,cou,env_list,c


def calc_true_envelope_spectral(signal_inp,fft_size,thresh,num_coeff,stopping_iters):
	"""
	Returns the Spectral Envelope based on the iterative version of the Windowed Cepstral 'Liftering' method

	Iteratively pushes the windowed liftered cepstral envelope towards the 'true' envelope

	Parameters
	----------
	signal_inp : np.array
	    numpy array containing the audio signal in the spectral domain with log magnitude(inp = log10(|X|))
	fft_size : integer(even)
	    FFT Size
	window : string
		Window function
	thresh : float
		The stopping criteria for the final envelope(Stop when final lies within init +- thresh), dB value
	num_coeff : integer
		Number of coefficients to consider for the cepstrum
	stopping_iters : integer
		Upper bound on number of iterations(if no convergence)


	Returns
	-------
	spectral_envelope : np.array
	    Returns the spectral envelope computed by the true method
	cou : int
		Number of iterations required to converge
	env_list : list(np.arrays)
		List containing the spectral envelope for each iteration 

	References
	----------
	.. [1] Röbel, Axel, and Xavier Rodet. "Efficient spectral envelope estimation and its application to pitch shifting and envelope preservation." International Conference on Digital Audio Effects. 2005.

	"""

	A_ip1 = signal_inp
	A_0 = A_ip1
	env_list = []

	# Threshold array
	thresh_arr = thresh*np.ones(fft_size)
	
	cou = 0
	while(True):

		
		V_i,w,c = ceps_envelope(10**(A_ip1),fft_size,'hann',44100,100,num_coeff,1,1)
		# c_im1 = c
		A_ip1 = np.where((A_ip1 > V_i),A_ip1,V_i)
		
		# c_p = np.real(np.fft.ifft(A_ip1))
		# Computing the In-Band and Out-of-Band Energies
		# E_i = np.linalg.norm((c_p - c_im1)[:num_coeff])**2
		# E_o = np.linalg.norm((c_p - c_im1)[num_coeff + 1:fft_size//2 + 1])**2
		# Computing the Adaptive weighting factor
		# adaptive_lambda = ((E_i + E_o)/E_i)
		# adaptive_lambda = 1
		# c_p = adaptive_lambda*(c_p - c_im1) + c_im1
		# A_ip1 = np.real(np.fft.fft(c_p))

		# print('iteration : ',cou + 1)
		cou = cou + 1

		env_list.append(A_ip1)

		# Stopping Criteria
		if((((A_0 - V_i) <= thresh_arr).all()) or (cou >= stopping_iters)):
			Vf = V_i
			break

	return Vf,cou,env_list

def lpc_to_lsf(lpc_coeffs):
	"""
	Returns the Line Spectral Frequencies(derived from the LPC) of the input frame(Same number of LSF's as LPC's)

	Parameters
	----------
	lpc_coeffs : np.array
	    numpy array containing the lpc coefficients

	Returns
	-------
	lsf_coeffs : np.array
	    Returns the LSF coefficients

	References
	----------
	.. [1]. Kondoz, A. M. Digital speech. Second Edition, 2004.(Pg. 95) 
	"""

	l = lpc_coeffs

	# Extracting the Sum and Difference Polynomials from the LPC coefficients
	A = [1]
	B = [1]
	p = l.shape[0] - 1
	alpha = l[1:]
	for k in range(1,p + 1):
		A.append((alpha[k - 1] - alpha[p - k]) + A[k-1])
		B.append((alpha[k - 1] + alpha[p - k]) - B[k-1])

	A = np.asarray(A)
	B = np.asarray(B)

	# Extracting the Roots of the Polynomial, and obtaining the arguments
	rr_A = np.roots(A)
	rr_B = np.roots(B)

	# Sorting the angles
	ws = np.sort(np.append(np.angle(rr_A),np.angle(rr_B)))

	# Keeping only the positive angles(0 <= w <= pi){This is effectively the LSF frequencies(normalized)}
	lsfs = ws[ws>=0]

	lsf_coeffs = lsfs

	return lsf_coeffs

def lsf_to_lpc(lsf_coeffs):
	"""
	Returns the LPC coefficients given the Line Spectral Frequencies

	Parameters
	----------
	lsf_coeffs : np.array
		LSF's as calculated by the funtion lpc_to_lsf()

	Returns
	-------
	lpc_coeffs : np.array
	    Returns the LPC coefficients

	References
	----------
	.. [1]. Kondoz, A. M. Digital speech. Second Edition, 2004.
	"""

	lsfs = lsf_coeffs
	# Obtain the even roots(corresponding to the Sum Polynomial P) and odd roots(Corresponding to the DIfference Polynomial Q)
	# Odd(Q)
	wB_r = lsfs[::2]
	# Even(P)
	wA_r = lsfs[1::2]

	# Append the conjugated roots to the above and form the complete coefficients
	roots_A = np.append(np.exp(wA_r*1j),np.exp(wA_r*-1j))
	roots_B = np.append(np.exp(wB_r*1j),np.exp(wB_r*-1j))

	# Form the polynomial from the roots
	P = np.poly(roots_A)
	Q = np.poly(roots_B)

	# Obtaining the Coefficients from the definition of the polynomial split
	lpc_coeffs = 0.5*(np.convolve(P,[1,-1]) + np.convolve(Q,[1,1]))[:-1]

	return lpc_coeffs

def cc_to_lpc(cepstral_coeffs,lpc_order):
	"""
	Returns the LPC Coefficients given the Cepstral coefficients and the lpc_order. Uses the recursive method to calculate.

	Parameters
	----------
	cepstral_coeffs : np.array
		Cepstral Coefficient array
	lpc_order : integer
		Order of cepstral coefficients to keep

	Returns
	-------
	lpc_coeffs : np.array
	    Returns the LPC coefficients

	References
	----------
	.. [1]. https://in.mathworks.com/help/dsp/ref/lpctofromcepstralcoefficients.html
	"""

	M = lpc_order
	# Defining the lpc array
	lpc_coeffs = [1]

	# Starting the recursion
	for m in range(1,M+1):
		temp_sum = 0
		for k in range(1,m-1):
			temp_sum = temp_sum + (m-k)*cepstral_coeffs[m-k]*lpc_coeffs[k]
		temp_sum = temp_sum/m
		lpc_coeffs.append(-cepstral_coeffs[m] - temp_sum)

	lpc_coeffs = np.asarray(lpc_coeffs) 
	
	return lpc_coeffs


def lpc_to_cc(lpc_coeffs,ceps_order,lpc_order):
	"""
	Returns the Cepstral Coefficients given the LPC coefficients and the cepstral order. Uses the recursive method to calculate.

	Parameters
	----------
	lpc_coeffs : np.array
		LPC's as calculated by the funtion cc_to_lpc()
	ceps_order : integer
		Order of cepstral coefficients to keep
	lpc_order : integer
		Order of lpc coefficients available

	Returns
	-------
	ceps_coeffs : np.array
	    Returns the Cepstral coefficients

	References
	----------
	.. [1]. https://in.mathworks.com/help/dsp/ref/lpctofromcepstralcoefficients.html
	"""

	# First Cepstral Coefficient set to 0(IF ANYTHING DON't WORK, CHECK THIS!!!!)
	ceps_coeffs = [0]
	N = ceps_order
	p = lpc_order
	for m in range(1,N):
		temp_sum = 0
		if(m <= p):
			for k in range(1,m-1):
				temp_sum = temp_sum + -1*(m-k)*ceps_coeffs[m-k]*lpc_coeffs[k]
			temp_sum = temp_sum/m
			ceps_coeffs.append(-lpc_coeffs[m] + temp_sum)
		else:
			for k in range(1,p):
				temp_sum = temp_sum + -1*(m-k)*ceps_coeffs[m-k]*lpc_coeffs[k]
			temp_sum = temp_sum/m
			ceps_coeffs.append(temp_sum)

	ceps_coeffs = np.asarray(ceps_coeffs)

	return ceps_coeffs










