# Neural Language Model (4-gram)

This project contains the implementation of a 4-gram language model using a Multilayer-Perceptron in Python. For more details on a Neural Language Model please read: http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf

## Data Preprocessing
The data files contain one sentence per line. A vocabulary is created by reading the training file and is capped at 8000 words. Eaxh sentence start with a START and ends with an END sign. The out of vocabulary words are replaced by UNK.

## Files

1. nn_lm_linear.py 
> Implementation of a Neural Network language Model with Linear Hidden Layer.
2. nn_lm_tanh.py 
> Implementation of a Neural Network language Model with tanh Hidden Layer.

## Hyperparameters
They are given as command line arguments as follows: 

* out_dir: Directory to store the models
* log_file: File for saving output logs
* hidden_dim: Number of hidden units
* input_dim: Embedding Size
* lr: Learning rate used for gradient updates
* batch_szie: Batch Size
* epochs: Number of iterations to be run for the dataset
* train: Training data file
* valid: Validation data file
* test: Test data file


__*Example run command for running linear language model:*__
> `python nn_lm_linear.py --out_dir models/ --input_dim 16 --hidden_dim 512 --lr 0.01 --batch_size 512 --epochs 130 --log_file logs/linear_512.log`

__*Example run command for running non-linear language model:*__
> `python nn_lm_tanh.py --out_dir models/ --input_dim 16 --hidden_dim 512 --lr 0.01 --batch_size 512 --epochs 130 --log_file logs/non_linear512.log`