# Parametric Representation of the Audio

The code in the directory obtains the parametric representation of the data. 

1. *data_processing.py* - This contains code to do the following-  
	1. Load the data from the folder specified (Can give as input the Train/Test Directories obtained in the data-loading code)
	2. Obtain the sustain part of the audio by energy thresholding.
	3. HpR Model on the sustain audio. The output is the (Magnitude,Harmonic) pairs + residual. We neglect the residual for now.
	![Parametric_1_HpR](https://www.ee.iitb.ac.in/student/~krishnasubramani/ex/pm_1.png)
	4. TAE - We obtain the TAE envelope from the approximate linearly interpolated spectrum obtained from the above (Magnitude,Harmonic) pairs.   
	![Parametric_2_TAE](https://www.ee.iitb.ac.in/student/~krishnasubramani/ex/tae_snap_git.png)  
	The TAE is the smooth function which we can sample at multiples of the fundamental frequency to obtain the corresponding magnitudes. It is described by the real cepstral coefficients.   
	5. Dump file - Save the cepstral coefficients corresponding to the TAE + the MIDI pitch into a pickle dump.

As a comparison with spectral reconstruction (literature based) methods, the following code is also included-  

2. *data_processing\_fft.py* - Instead of the parametric representation, we just save the framewise magnitude spectra into a pickle dump.

To play around with the parameters and directory names, change the appropriate variable values in the code (comments have been added next to the control variables)

---

To test the working of the above parametric modeling, we have also written code that 'inverts' the parametric model to obtain the audio. 

1. *pp_test.py* - This obtains the parametric reconstruction of the input audio. This is done by,  
	1. Sampling the harmonic amplitudes from the TAE envelope.
	2. Performing a Sinusoidal Model reconstruction from the obtained (Magnitude,Harmonic) pairs.
2. *pp_test\_fft.py* - This 'inverts' the magnitude spectrogram using the griffin lim algorithm as implemented in librosa.

Both the above files work on the pickle dumps obtained in the processing step, and output the audio for all the files in the dump. For proper inversion, ensure that the parameters values are consistent during obtaining the parameters and the inversion.   

To analyze the number of frames obtained per note, an additional Jupyter Notebook - *Preprocess-Stats.ipynb* displays the number of frames for each note in the chosen octave. This can be helpful to choose the number of frames to train the model on.




