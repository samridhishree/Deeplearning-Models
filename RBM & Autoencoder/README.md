# Restricted Boltzmann Machine & Autoencoder

This project contains an implementation of the Contrastive Divergence (CD) algorithm for training a Restricted Boltzmann Machine (RBM). It also implemnts an Autoencoder which is compared to the RBM. These routines are further used as pretraining modules on a multi-layer neural network and the results are compared to the performance of the network without pretraining.

## Data
It used the MNIST dataset, that contain images of size 28 b 28. These images are transformed to be represented as a vector of dimension 784 by listing all the pixel values in raster scan order. Each image has a correspoinding label value ranging from 0,....9. It identifies the digit in the image. 

## Format of the data
digitstrain.txt contains 3000 lines. Each line contains 785 numbers (comma delimited): the first 784 real-valued numbers correspond to the 784 pixel values, and the last number denotes the class label: 0 corresponds to digit 0, 1 corresponds to digit 1, etc. digitsvalid.txt and digitstest.txt contain 1000 and 3000 lines and use the same format as above.


## Files

1. rbm.py 
> Implementation of a Restricted Boltzmann Machine.
2. autoencoder.py 
> Creates an autoencoder with a command line argument to include noise
3. two_layer_neural_network_batch_norm.py 
> Creates a two layer neural network with batch norm (used as the target network for the pretrained output)

## Hyperparameters
They are given as command line arguments as follows:

* out_dir: Directory to store the models
* hidden_dim: Number of hidden units
* lr: Learning rate used for gradient updates
* k: Number of giibs steps to take for an RBM
* dropout: Percentage of noise to be introduced for a denoising autoencoder
* epochs: Number of iterations to be run for the dataset
* train: Training data file
* valid: Validation data file
* test: Test data file


__*Example run command for running RBM:*__
`python -u rbm.py --out_dir 'hidden_rbm/' --hidden_dim 100 --lr 0.01 --k 1 --epochs 200 >> run1.out`

__*Example run command for running autoencoder:*__
`python -u autoencoder.py --out_dir hidden_denoise/ --hidden_dim 100 --lr 0.01 --dropout 0.0 --epochs 200 >> run2.out`

__*Example run command for denoising autoencoder:*__
`python -u autoencoder.py --out_dir hidden_denoise/ --hidden_dim 100 --lr 0.01 --dropout 0.1 --epochs 200 >> run3.out`