The hyper parameter are given as command line arguments. There are 2 files included:

1. rbm.py - Implementation of a Restricted Boltzmann Machine.
2. autoencoder.py - Creates an autoencoder with a command line argument to include noise
3. two_layer_neural_network_batch_norm.py - Creates a two layer neural network with batch norm

The command line arguments are as follows:

--out_dir: Directory to store the models
--hidden_dim: Number of hidden units
--lr: Learning rate used for gradient updates
--k: Number of giibs steps to take for an RBM
--dropout: Percentage of noise to be introduced for a denoising autoencoder
--epochs: Number of iterations to be run for the dataset
--train: Training data file
--valid: Validation data file
--test: Test data file


Example run command for running RBM:
python -u rbm.py --out_dir 'hidden_rbm/' --hidden_dim 100 --lr 0.01 --k 1 --epochs 200 >> run1.out

Example run command for running autoencoder:
python -u autoencoder.py --out_dir hidden_denoise/ --hidden_dim 100 --lr 0.01 --dropout 0.0 --epochs 200 >> run2.out

Example run command for denoising autoencoder:
python -u autoencoder.py --out_dir hidden_denoise/ --hidden_dim 100 --lr 0.01 --dropout 0.1 --epochs 200 >> run3.out