# non-Gaussian
The code of estimating the non-Gaussian groundwater conductivity parameter
# GAN-ILUES or GAN-OANW-ILUES?
The code 'GAN-OANW-ILUES' contains two frameworks:GAN-ILUES and GAN-OANW-ILUES. We can run "using surrogate model" to operate the GAN-OANW-ILUES and "no surrogate modle" for the GAN-ILUES.
# Deep learning method
## The generation model of the hydraulic conductivity field
The trained generator used here is "netG_epoch_27.pth", which is upload already. We can load the trained model by running the code in line 106.
## The surrogate model OANW
The surrogate model OANW "model_epoch198.pth". We can load the model by running code in line 365.
# Iterative Local Updating Ensemble Smoother
The data assimilation method used here is ILUES, which is compiled by Matlab 2021b. 
