# PyTorch implementation of the Conditional Variational Autoencoder (CVAE)

The code for our network is inspired from <a href="https://github.com/timbmg/VAE-CVAE-MNIST" target="_blank">timbmg's</a> implementation of the same. The network is summarized in the following figure:   
![Network Architecture](https://www.ee.iitb.ac.in/student/~krishnasubramani/ex/net_arch.png)

Network Source Code-  

1. simple_ae.py - Autoencoder (Deterministic feed-forward). 
2. vae_krishna.py - Conditional Variational Autoencoder. There is a $\beta$ weight term between the Variational loss and the reconstruction loss which can be changed accordingly (We use $\beta = 0.1$)
3. dataset.py - Our custom PyTorch Data Loader. It has been written to take as input the output of the Pickle Dump (generated as the output of the Parametric/Spectral Modeling). It loads the data from the dump into the dataloader.

For both networks, checkout the code for additional parameters (number of layers, latent space dimensionality, CUDA availability etc). We use linear fully-connected layers with leaky-ReLU activations   

Network Architecture-   

Trained on the parametric representation
1. synth_vae.py - CVAE
2. ae_synth.py - AE

Trained on the spectrum
1. synth_vae_fft.py - CVAE
2. ae_synth_fft.py - AE

All the above networks save the weights of the trained network (and the network description) into a PyTorch .pth file (into the specified directory in the specific code)

---

To evaluate the network working we have written network\_data\_recon.py and network\_data\_recon\_fft.py which performs 2 things-     
1. Reconstructs the input audio using the network
2. Compute the MSE (averaged across all the input pitches)

The file requires the PyTorch network weights of both CVAE and AE to run (sorry for the inefficiently written code, hope to improve this!). When the code is run, it will ask you to select the .pth file for both CVAE and AE, and also specify the directory to save the network generated audio (along with the MSE plot for each pitch)   
For the parametric representation, the audio is obtained by sampling the envelope and performing a sinusoidal reconstruction. For the magnitude spectrum reconstruction, Griffin-Lim is used to invert the magnitude spectrogram to obtain the audio.
