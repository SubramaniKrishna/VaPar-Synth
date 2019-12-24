# VaPar Synth - A Variational Parametric Model for Audio Synthesis

### Krishna Subramani<sup>1</sup>, Preeti Rao<sup>1</sup>, Alexandre D'Hooge<sup>2</sup>
### <sup>1</sup>IIT Bombay, <sup>2</sup>ENS Paris-Saclay

<a href="https://www.ee.iitb.ac.in/student/~krishnasubramani/data/icassp_paper.pdf" target="_blank">Paper</a> 	/	 <a href="https://www.ee.iitb.ac.in/student/~krishnasubramani/icassp2020.html" target="_blank">Accompanying Webpage</a> 	/	<a href="https://www.ee.iitb.ac.in/student/~krishnasubramani/data/vapar.bib" target="_blank">BibTeX</a>

This repository contains the code for **VaPar Synth**, a Conditional Variational Autoencoder trained on a source-filter inspired parametric representation.  
<!-- summarized below in the figure, ![Network Architecture](https://www.ee.iitb.ac.in/student/~krishnasubramani/ex/net_arch.png) -->

For the necessary libraries/prerequisites, please use conda/anaconda to create an environment (from the environment.yml file in this repository) with the command   
~~~
	conda env create -f environment.yml
~~~
Also install <a href="https://github.com/MTG/sms-tools" target="_blank">SMS-Tools</a> in the same environment. With these, all the code in the repository can be run inside this environment by activating it.  

The repository contains code to do the following (One directory for each)-

1. [Data Loading](./Data_Loading/README.md): We use the <a href="https://zenodo.org/record/820937#.XgB01HUzZhE" target="_blank">good-sounds</a> dataset. 
2. [Parametric](./Parametric/README.md): Obtaining the parametric representation of the audio.   
	1. The parametric representation used utilizes the **True Amplitude Envelope Estimation (TAE)** algorithm on top of the Harmonic plus Residual model.   
	2. We have implemented the code for **TAE** using the algorithm described <a href="https://hal.archives-ouvertes.fr/hal-01161334" target="_blank">here</a> . To the best of our knowledge, there is no open-source implementation (in Python) for this algorithm, so we make our code available to the community to use it and build upon/improve it. The algorithm is present in the func\_envs.py file under the [extra\_dependencies](./extra_dependencies/README.md) directory. We have also demonstrated the algorithms usage in a Jupyter Notebook in the same directory. 
3. [Network](./Network/README.md):
	

