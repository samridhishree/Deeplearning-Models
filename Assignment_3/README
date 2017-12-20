The hyper parameter are given as command line arguments. There are 2 files included:

1. nn_lm_linear.py - Implementation of a Neural Network language Model with Linear Hidden Layer.
2. nn_lm_tanh.py - Implementation of a Neural Network language Model with tanh Hidden Layer.

The command line arguments are as follows:

--out_dir: Directory to store the models
--log_file: File for saving output logs
--hidden_dim: Number of hidden units
--input_dim: Embedding Size
--lr: Learning rate used for gradient updates
--batch_szie: Batch Size
--epochs: Number of iterations to be run for the dataset
--train: Training data file
--valid: Validation data file
--test: Test data file


Example run command for running linear language model:
python nn_lm_linear.py --out_dir models/ --input_dim 16 --hidden_dim 512 --lr 0.01 --batch_size 512 --epochs 130 --log_file logs/linear_512.log

Example run command for running non-linear language model:
python nn_lm_tanh.py --out_dir models/ --input_dim 16 --hidden_dim 512 --lr 0.01 --batch_size 512 --epochs 130 --log_file logs/non_linear512.log