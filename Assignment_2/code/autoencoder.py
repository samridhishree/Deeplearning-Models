
# coding: utf-8

# In[1]:


import os
import sys
import argparse
import cPickle as pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# In[2]:


'''
Creates a numpy matrix of x, no labels
'''
def ParseInputData(input_file):
    data = np.loadtxt(input_file, delimiter=',', usecols=range(0,784))
    return data

def create_images(weight):
    row, col = weight.shape
    print row
    print col
    images = []
    for c in range(col):
        image = np.reshape(weight[:,c],(28,28))
        images.append(image)
    return images

# Plot the 100 images in a 10X10 table
def plot_images(images):
    fig = plt.figure()
    #images = list(images)
    print len(images)
    for x in range(10):
        for y in range(10):
            ax = fig.add_subplot(10, 10, 10*y+x+1)
            ax.matshow(images[10*y+x], cmap = matplotlib.cm.binary)
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
    return plt


# In[42]:


#Neural Network class
class NeuralNetwork(object):
    def __init__(self, input_dim, hidden_dim, output_dim):
        '''
        input_dim: No. of input nodes (dimensionality of input data)
        hidden_dim: No. of neurons in the hidden layer
        output_dim: No. of neurons in the output layer (should be same as input_dim)
        '''
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        #self.batch_size = batch_size
        
        # Initialize the weights and biases.
        #seed = np.sqrt(6/(float)(self.input_dim + self.hidden_dim))
        #self.w = np.random.uniform(-seed, seed, (self.input_dim, self.hidden_dim))
        self.w = np.random.normal(0, 0.1, (self.input_dim, self.hidden_dim))
        self.b1 = np.zeros((1, self.hidden_dim))
        self.b2 = np.zeros((1, self.output_dim))
        
        # Initialize the intermediate activations to be used in backprop. Set to 1
        self.a1 = np.array([])
        self.out = np.array([])
        

    # Forward activation function - computes sigmoid(x)
    def sigmoid(self, x):
        #print "in sigmoid, x.shape = ", x.shape
        return 1.0/(1.0 + np.exp(-x))

    # Backward propogation helper, calculates the sigmoid derivative. Input is sigmoid(x).
    def sigmoid_derivative(self, sig_x):
        return (sig_x * (1.0-sig_x))

    
    # Function to perform feedforward computations - stochastic
    def feed_forward(self, input_X):
        z1 = input_X.dot(self.w) + self.b1
        #vfunc = np.vectorize(self.sigmoid, otypes=[np.float])
        self.a1 = self.sigmoid(z1)
        z2 = (self.a1).dot(self.w.T) + self.b2
        return self.sigmoid(z2)
    
    # Function to perform backward propagation. Returns the loss - stochastic
    def back_propogation(self, input_X, dropped_x, eta):
        '''
        eta: Learning Rate
        '''
        delta1 = self.out - input_X
#         print "delta1 shape = ", delta1.shape
#         print "self.out shape = ", self.out.shape
#         print "input_X shape = ", input_X.shape
#         print "self.w shape = ", self.w.shape
        # Sigmoid derivative on the first layer activations
        #vfunc = np.vectorize(self.sigmoid_derivative, otypes=[np.float])
        sig_der = self.sigmoid_derivative(self.a1)
        temp = delta1.dot((self.w))
        delta2 = temp * sig_der
        dw1 = (dropped_x.T).dot(delta2)
        dw2 = (self.a1.T).dot(delta1)
        dw = dw1 + dw2.T
        #print "shape of delta2 (db1) = ", delta2.shape
        db1 = np.sum(delta2, axis=0, keepdims=True) 
        db2 = np.sum(delta1, axis=0, keepdims=True)
        #print "db1 = ", db1
        #print "db2 = ", db2
        self.w = self.w - (eta * dw)
        self.b1 = self.b1 - (eta * db1)
        self.b2 = self.b2 - (eta * db2)
        
        # Calculate the loss for this batch
        loss = self.CalculateLoss(self.out, input_X)
        return loss
    
    # Calculates the loss with the current values of the weight matrices
    def CalculateLoss(self, pred_x, true_x):
        loss = np.sum(-(true_x*np.log(pred_x)+(1-true_x)*np.log(1-pred_x)))
        return loss/float(len(pred_x))

    def CalculateValidLoss(self, valid_data):
        total_val_loss = 0.0
        for val_x in valid_data:
            pred_val_x = self.feed_forward(val_x)
            val_loss = self.CalculateLoss(pred_val_x, val_x)
            total_val_loss += val_loss
        avg_val_loss = total_val_loss/float(len(valid_data))
        return avg_val_loss
    
    # Perform Training
    def train(self, train_data, valid_data, dir_to_save, epochs, eta, dropout):
        # eta: Learning Rate
        # targets are in a one hot representation format for their slice
        model = {}
        losses = []
        valid_losses = []
        min_val_loss = np.inf
        optimal = {}

        for epoch in range(epochs):
            print " Epoch : ", epoch
            # Random shuffle batches
            np.random.shuffle(train_data)
            #np.random.shuffle(train_data)
            total_loss = 0.0
            #print "Total Instances : ", total_instances
            for input_X in train_data:
                input_X = input_X.reshape(1, input_X.shape[0])
                mask = np.random.binomial(1, (1-dropout), size=input_X.shape)
                dropped = input_X * mask
                self.out = self.feed_forward(dropped)
                loss = self.back_propogation(input_X, dropped, eta)
                total_loss += loss
            avg_loss = total_loss/(float)(len(train_data))
            losses.append(avg_loss)
            print "Epoch : ", epoch, ", Avg training loss : ", avg_loss
            valid_loss = self.CalculateValidLoss(valid_data)
            valid_losses.append(valid_loss)
            print "Epoch : ", epoch, ", Avg validation loss : ", valid_loss
            if valid_loss < min_val_loss:
                optimal['w'] = self.w
                optimal['b1'] = self.b1
                optimal['b2'] = self.b2
                min_val_loss = valid_loss

            # Save generated samples after every five epochs
            # if epoch % 5 == 0:
            #     print "Generating samples at epoch = ", epoch
            #     temp_model = {}
            #     temp_model['w'] = self.w
            #     temp_model['b1'] = self.b1
            #     temp_model['b2'] = self.b2
            #     filename = 'model_epoch_' + str(epoch) + '_val_loss_' + str(valid_loss) + '.pickle'
            #     pickle_file = os.path.join(dir_to_save, filename)
            #     print "Saving file = ", pickle_file
            #     with open(pickle_file, 'wb') as p_pickle:
            #         pickle.dump(model, p_pickle)

            #     # Visualize weights
            #     w_viz = create_images(self.w)
            #     w_plt = plot_images(w_viz)
            #     weight_file = os.path.join(dir_to_save, 'weight_filters_epoch_' + str(epoch) + '.png')
            #     plt.savefig(weight_file)

        
        # Create model dictionary
        min_valid = np.argmin(valid_losses)
        print "Minimum validation loss at epoch : ", min_valid , " with a value of ", np.min(valid_losses)
        model['w'] = self.w
        model['b1'] = self.b1
        model['b2'] = self.b2
        model['train_loss'] = losses
        model['valid_loss'] = valid_losses
        model['min_valid_loss'] = np.min(valid_losses)
        model['optimal'] = optimal
        return model


# In[43]:

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, default='output',
                        help = 'Directory to store the models and plots')
    parser.add_argument('--hidden_dim', type=int, default=100,
                        help = 'Number of hidden units')
    parser.add_argument('--lr', type=float, default=0.01,
                        help = 'Learning rate used for gradient updates')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help = 'Noise rate for the input vector')
    parser.add_argument('--epochs', type=int, default=200,
                        help = 'Number of iterations to be run for the dataset')
    parser.add_argument('--train', type=str, default='data/digitstrain.txt',
                        help = 'Training data file')
    parser.add_argument('--valid', type=str, default='data/digitsvalid.txt',
                        help = 'Validation data file')
    args = parser.parse_args()

    # Hard Coded Values
    in_dim = 784
    out_dim = 784
    print(sys.argv)

    train_data = ParseInputData(args.train)
    valid_data = ParseInputData(args.valid)


    subdir = 'autoenc_dim_' + str(args.hidden_dim) +'_gen_samples'
    dir_to_save = os.path.join(args.out_dir, subdir)
    try:
        os.makedirs(dir_to_save)
    except:
        pass
    nn = NeuralNetwork(input_dim=in_dim, hidden_dim=args.hidden_dim, output_dim=out_dim)
    model = nn.train(train_data=train_data, valid_data=valid_data, dir_to_save=dir_to_save, 
                  epochs=args.epochs, eta=args.lr, dropout=args.dropout)

    subdir = 'models'
    filename = 'model_lr=' + str(args.lr) + '_hidden=' + str(args.hidden_dim) + 'valid_min=' + str(model['min_valid_loss']) + '.pickle'
    dir_to_save = os.path.join(args.out_dir, subdir)

    try:
        os.makedirs(dir_to_save)
    except:
        pass

    pickle_file = os.path.join(dir_to_save, filename)
    print "Saving file = ", pickle_file
    with open(pickle_file, 'wb') as p_pickle:
        pickle.dump(model, p_pickle)

    # Genrating plots - loss curves, weights visualtions and samples visualization
    subdir = 'Autoenc_hidden_' + str(args.hidden_dim) +'_plots'
    dir_to_save = os.path.join(args.out_dir, subdir)
    
    try:
        os.makedirs(dir_to_save)
    except:
        pass

    # Loss curves
    plt.close()
    plt.plot(range(args.epochs),model['train_loss'], label="Train")
    plt.plot(range(args.epochs),model['valid_loss'], label="Valid")
    plt.legend()
    loss_file = os.path.join(dir_to_save, 'loss_curves.png')
    plt.savefig(loss_file)

    # Weight Visualization
    plt.close()
    w_viz = create_images(model['optimal']['w'].T)
    w_plt = plot_images(w_viz)
    weight_file = os.path.join(dir_to_save, 'weight_filters.png')
    plt.savefig(weight_file)

if __name__ == "__main__":
    main()

