# Non Library Code/Dependencies

1. <u>func_envs.py</u> - This is the python file which contains companion functions to do the following - 
	1. Extract Envelopes - ceps_envelope(), lpc_envelope(), calc_true_envelope(), calc_true_envelope_spectral()
	2. Miscallaneous helper functions - Coefficient Conversions (LPC <-> CC <-> LSF)

2. <u>models</u> - SMSTools. Code to perform Sinusoidal, Harmonic, Residual analysis

3. <u>morphing.py</u> - Functions to morph two audio by linearly interpolating in the appropriate space (lpc, cc, lsf, spectrum etc.)

There is also a Jupyter notebook TAE_demo.ipynb which demonstrates the usage of the various envelope extraction functions.