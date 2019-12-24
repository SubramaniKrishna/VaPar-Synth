"""
Script for a Autoencoder(Naive) to generate(synthesize) sounds.
"""

# Dependencies
import torch
import torch.nn as nn


# Defining the Classes corresponding to the Encoder and Decoder

class AE_Enc(nn.Module):
	"""
	Class specifying the conditional encoder architecture(X -> Code)
	
	Constructor Parameters
	----------------------
	layer_dims : list of integers
		List containing the dimensions of consecutive layers(Including input layer{dim(X)} and excluding latent layer)
	latent_dims : integer
		Size of latent dimenstion(default = 2)
	"""

	# Defining the constructor to initialize the network layers and activations
	def __init__(self, layer_dims, latent_dims):
		super().__init__()
		self.layer_dims = layer_dims
		self.latent_dims = latent_dims

		# Initializing the Model as a torch sequential model
		self.ENC_NN = nn.Sequential()

		# Currently using Linear layers with ReLU activations(potential hyperparams)
		# This Loop defines the layers just before the latent space (input(X) -> layer[0] -> layer[1] .... -> layer[n])
		for i, (in_size, out_size) in enumerate(zip(layer_dims[:-1], layer_dims[1:])):
			self.ENC_NN.add_module(name = "Encoder_Layer_{:d}".format(i), module = nn.Linear(in_size, out_size))
			self.ENC_NN.add_module(name = "Activation_{:d}".format(i), module = nn.LeakyReLU())

		# Convert Layer to the Intermediary code
		self.ENC_NN.add_module(name = "Code_Linear", module = nn.Linear(layer_dims[-1], latent_dims))


	def forward(self, x):
		"""
		Forward pass of the encoder to obtain the latent space parameters

		Inputs
		------
		x : torch.tensor
			Input tensor
		"""
		# Forward pass till the n-1'th layer
		code = self.ENC_NN(x)

		return code


class AE_Dec(nn.Module):
	"""
	Class specifying the conditional decoder architecture(Code -> X')
	
	Constructor Parameters
	----------------------
	layer_dims : list of integers
		List containing the dimensions of consecutive layers(Including latent dimension{dim(X)} and including final layer, which should be same size as input)
	"""

	# Defining the constructor to initialize the network layers and activations
	def __init__(self, layer_dims):
		super().__init__()
		self.layer_dims = layer_dims

		# Initializing the Model as a torch sequential model
		self.DEC_NN = nn.Sequential()

		# Currently using Linear layers with ReLU activations(potential hyperparams)
		# This Loop defines the layers after the latent space(latent space -> layer[0] -> layer[1] .... -> layer[n] -> Reconstructed output)
		# <Point to note>, the final layer should allow negative values as well. Also, the inputs should be scaled before feeding to the network
		# to restrict them to (-1,1), and then use an appropriate activation(like tanh) to get the output in the range(-1,1) abd rescale it back.
		for i, (in_size, out_size) in enumerate(zip(layer_dims[:-1], layer_dims[1:])):
			self.DEC_NN.add_module(name="Decoder_Layer{:d}".format(i), module=nn.Linear(in_size, out_size))
			if i + 1 < len(layer_dims) - 1:
				self.DEC_NN.add_module(name="Activation{:d}".format(i), module=nn.LeakyReLU())
			else:
				self.DEC_NN.add_module(name="Reconstruct_LReLU", module=nn.LeakyReLU(negative_slope = 0.1))


	def forward(self, code):
		"""
		Forward pass of the decoder to obtain the reconstructed input
		
		Inputs
		------
		code : torch.tensor
			Latent variable
		"""

		# Reconstruct
		x_recon = self.DEC_NN(code)

		return x_recon


# Overall AE combining the Encoder and Decoder
class AE_synth(nn.Module):
	"""
	Class defining the autoencoder by combining the Encoder and Decoder
	X -> Encoder -> Code -> Decoder -> X'
	
	Constructor Parameters
	----------------------
	device : 'cuda' or 'cpu'
		Where to run the optimization
	layer_dims_enc : list of integers
		List containing the dimensions of consecutive layers for the encoder(Including input layer{dim(X)} and excluding latent layer)
	layer_dims_dec : list of integers
		List containing the dimensions of consecutive layers for the decoder(Including latent dimension{dim(X)} and including final layer, which should be same size as input)
	latent_dims : integer
		Size of latent dimenstion(default = 2)
	"""

	# Constructor defining the architecture
	def __init__(self, device, layer_dims_enc, layer_dims_dec, latent_dims = 2):
		super().__init__() #Understand more about how super works!

		self.layer_dims_enc = layer_dims_dec
		self.layer_dims_dec = layer_dims_dec
		self.latent_dims = latent_dims
		self.device = device

		# Defining the Encoder and Decoder architecture by calling the previously defined classes
		# Encoder
		self.main_ENC = AE_Enc(layer_dims = layer_dims_enc, latent_dims = latent_dims)
		# Decoder
		self.main_DEC = AE_Dec(layer_dims = layer_dims_dec)


	# Forward pass
	def forward(self, x):
		"""
		Forward pass of the Autoencoder
		
		Inputs
		------
		x : torch.tensor
			Input tensor
		"""

		code = self.main_ENC(x)
		recon_x = self.main_DEC(code)

		return recon_x, code