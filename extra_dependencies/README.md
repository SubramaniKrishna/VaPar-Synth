# Additional Code/Dependencies

1. *func_envs.py* - This is the python file which contains companion functions to do the following - 
	1. Extract Envelopes - ceps_envelope(), lpc_envelope(), calc\_true\_envelope(), calc\_true\_envelope\_spectral()
	2. Miscallaneous helper functions, Coefficient Conversions (LPC <-> CC <-> LSF) etc

2. *models* - SMSTools. Code to perform Sinusoidal, Harmonic, Residual analysis

3. *morphing.py* - Functions to morph two audio by linearly interpolating in the appropriate space (lpc, cc, lsf, spectrum etc.)

There is also a Jupyter notebook *TAE_demo.ipynb* which demonstrates the usage of the various envelope extraction functions.