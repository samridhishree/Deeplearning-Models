# Multi-Layer Neural Network

This project contains an implementation for a multi-layer neural network in python. It implements different activation functions and the network backpropagation. It works on mini-batched training data and employs L2 regularization with momentum in the SGD training update. There is also an implementation for batch normalization. The error plots and the visulaization of the weights of the network can be found under the `plots/` directory. 

## Data
It used the MNIST dataset, that contain images of size 28 b 28. These images are transformed to be represented as a vector of dimension 784 by listing all the pixel values in raster scan order. Each image has a correspoinding label value ranging from 0,....9. It identifies the digit in the image. 

## Format of the data
digitstrain.txt contains 3000 lines. Each line contains 785 numbers (comma delimited): the first 784 real-valued numbers correspond to the 784 pixel values, and the last number denotes the class label: 0 corresponds to digit 0, 1 corresponds to digit 1, etc. digitsvalid.txt and digitstest.txt contain 1000 and 3000 lines and use the same format as above.

## Code Files

1. single_layer_nn.py 
> Creates a single layer neural network.
2. two_layer_neural_network.py 
> Creates a two layer neural network
3. two_layer_neural_network_batch_norm.py
> Creates a two layer neural network with batch norm

## Hyperparameters
They are given as command line arguments as follows:

### Single Layer
* out_dir: Directory to store the models
* hidden_dim: Number of hidden units
* lr: Learning rate used for gradient updates
* reg_param: Regularization Parameter for weight updates
* momentum: Momentum used for incremental parameter updates
* batch_size: Batch Size
* epochs: Number of iterations to be run for the dataset
* train: Training data file
* valid: Validation data file
* test: Test data file


### Extra parameters for two layer NN (in addition to above)
* layer1_dim: Number of hidden units in first layer
* layer2_dim: Number of hidden units in second layer')
* loss: Type of loss - 'sigmoid', 'relu' or 'tanh'

__*Example run command for running single layer network and piping the output to a file:*__
> `python -u single_layer_nn.py --out_dir 'models/one_layer/' --hidden_dim 100 --lr 0.1 --reg_param 0 --momentum 0 --batch_size 32 --epochs 200 *  run1.out`

__*Example run command for running two layer network and piping the output to a file:*__
> `python -u two_layer_neural_network.py --out_dir 'new_models/two_layer/' --layer1_dim 200.0 --layer2_dim 100.0 --lr 0.01 --reg_param 0.0 --momentum 0.0 --batch_size 32 --epochs 200 --loss sigmoid *  run2.out`

__*Example run command for two layer network with batchnorm and piping the output to a file:*__
> `python -u two_layer_neural_network_batch_norm.py --out_dir 'new_models/two_layer_bn/' --layer1_dim 200.0 --layer2_dim 100.0 --lr 0.01 --reg_param 0.0 --momentum 0.0 --batch_size 32 --epochs 200 *  run3.out`