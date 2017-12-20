
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


# In[7]:


'''
Get the data matrix
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

'''
Sigmoid of x
'''
def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))


# In[55]:


# RBM class
class RBM(object):
    def __init__(self, input_dim, hidden_dim, load_model, model_file):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.w = np.random.normal(0, 0.1, (self.hidden_dim, self.input_dim))
        self.b = np.zeros(self.hidden_dim)
        self.c = np.zeros(self.input_dim)

        if load_model == True:
            print "Loading the model and initilaizing the weights"
            model = pickle.load(open(model_file, 'rb'))
            self.w = model['w']
            self.b = model['b']
            self.c = model['c']

    
    '''
    Sample a bernoulli variable from the given probability
    '''
    def sample(self, p):
        return (np.random.binomial(1, p, size=p.shape))
    
    '''
    Gibbs Chain
    '''
    def gibbs_chain(self, x, k):
        for i in range(k):
            prob_h_x = sigmoid(self.b + (x).dot(self.w.T))
            h = self.sample(prob_h_x)
            prob_x_h = sigmoid(self.c + (h).dot(self.w))
            x = self.sample(prob_x_h)
        return x
    
    '''
    Generate sample by performing k steps
    '''
    def generation(self, x, k):
        for i in range(k):
            prob_h_x = sigmoid(self.b + (x).dot(self.w.T))
            h = self.sample(prob_h_x)
            prob_x_h = sigmoid(self.c + (h).dot(self.w))
            x = self.sample(prob_x_h)
        return prob_x_h
    
    def calculate_loss(self, x):
        prob_h_x = sigmoid(self.b + (x).dot(self.w.T))
        x_pred = sigmoid(self.c + (prob_h_x).dot(self.w))
        total_loss = np.sum(-x*np.log(x_pred) - (1-x)*np.log(1-x_pred))
        avg_loss = total_loss/float(len(x_pred))
        return avg_loss
    
    
    def train(self, train_data, valid_data, dir_to_save, lr=0.01, k=1, epochs=200):
        model = {}
        losses = []
        valid_losses = []
        min_val_loss = np.inf
        optimal = {}
        
        for epoch in range(epochs):
            np.random.shuffle(train_data)
            print " Epoch : ", epoch
            for x in train_data:
                x_cap = self.gibbs_chain(x, 1)
                prob_h_x = sigmoid(self.b + (x).dot(self.w.T))
                prob_h_cap_x = sigmoid(self.b + (x_cap).dot(self.w.T))
                
                grad_w = np.outer(prob_h_x, x) - np.outer(prob_h_cap_x, x_cap)
                grad_b = prob_h_x - prob_h_cap_x
                grad_c = x - x_cap
                
                self.w = self.w + lr*grad_w
                self.b = self.b + lr*grad_b
                self.c = self.c + lr*grad_c
                
            train_loss = self.calculate_loss(train_data)
            val_loss = self.calculate_loss(valid_data)
            losses.append(train_loss)
            valid_losses.append(val_loss)
            # Check and save validation loss and weights
            if val_loss < min_val_loss:
                optimal['w'] = self.w
                optimal['b'] = self.b
                optimal['c'] = self.c
                min_val_loss = val_loss
            print "Epoch : ", epoch, ", Avg training loss : ", train_loss
            print "Epoch : ", epoch, ", Avg validation loss : ", val_loss

            # Save generated samples after every five epochs
            if epoch % 5 == 0:
                print "Generating samples at epoch = ", epoch
                temp_model = {}
                temp_model['w'] = self.w
                temp_model['b'] = self.b
                temp_model['c'] = self.c
                filename = 'model_epoch_' + str(epoch) + '_val_loss_' + str(val_loss) + '.pickle'
                pickle_file = os.path.join(dir_to_save, filename)
                print "Saving file = ", pickle_file
                with open(pickle_file, 'wb') as p_pickle:
                    pickle.dump(temp_model, p_pickle)

                # Generate samples
                fig = plt.figure()
                for i in range(100):
                    x_init = np.random.uniform(0.0, 1.0, size=784)
                    #x_init = np.round(x_init)
                    x_sampled = self.generation(x_init, k=1000)
                    b = x_sampled.reshape(28,28)
                    plt.subplot(10, 10, i+1)
                    plt.axis('off')
                    plt.imshow(b, cmap='gray')
    
                generate_file = os.path.join(dir_to_save, 'generated_samples_epoch_' + str(epoch) + '_val_loss_' + str(val_loss) + '.png')
                plt.savefig(generate_file)
                plt.close()

        
        min_valid = np.argmin(valid_losses)
        print "Minimum validation loss at epoch : ", min_valid , " with a value of ", np.min(valid_losses)
        model['w'] = self.w
        model['b'] = self.b
        model['c'] = self.c
        model['train_loss'] = losses
        model['valid_loss'] = valid_losses
        model['min_valid_loss'] = np.min(valid_losses)
        model['optimal'] = optimal
        return model    


# In[56]:

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, default='output',
                        help = 'Directory to store the models and plots')
    parser.add_argument('--hidden_dim', type=int, default=100,
                        help = 'Number of hidden units')
    parser.add_argument('--lr', type=float, default=0.01,
                        help = 'Learning rate used for gradient updates')
    parser.add_argument('--k', type=int, default=1,
                        help = 'Number of gibbs sampling to be performed')
    parser.add_argument('--epochs', type=int, default=200,
                        help = 'Number of iterations to be run for the dataset')
    parser.add_argument('--train', type=str, default='data/digitstrain.txt',
                        help = 'Training data file')
    parser.add_argument('--valid', type=str, default='data/digitsvalid.txt',
                        help = 'Validation data file')
    parser.add_argument('--load_model', type=bool, default=False,
                        help = 'Is there a model to be loaded')
    parser.add_argument('--model_file', type=str, default='',
                        help='Pickle file to be loaded')
    args = parser.parse_args()

    # Hard Coded Values
    in_dim = 784
    print(sys.argv)

    train_data = ParseInputData(args.train)
    valid_data = ParseInputData(args.valid)

    subdir = 'RBM_K_' + str(args.hidden_dim) +'_gen_samples'
    dir_to_save = os.path.join(args.out_dir, subdir)
    try:
        os.makedirs(dir_to_save)
    except:
        pass
    rbm = RBM(input_dim=in_dim, hidden_dim=args.hidden_dim, load_model=args.load_model, model_file=args.model_file)
    model = rbm.train(train_data, valid_data, dir_to_save, args.lr, args.k, args.epochs)

    subdir = 'RBM_hidden_' + str(args.hidden_dim) +'_model'
    filename = 'model_lr=' + str(args.lr) + '_k=' + str(args.k) + 'valid_min=' + str(model['min_valid_loss']) + '.pickle'
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
    subdir = 'RBM_K_' + str(args.k) +'_plots'
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

    # Generate 100 images
    optimal_rbm = RBM(input_dim=in_dim, hidden_dim=args.hidden_dim, load_model=args.load_model, model_file=args.model_file)
    optimal_rbm.w = model['optimal']['w']
    optimal_rbm.b = model['optimal']['b']
    optimal_rbm.c = model['optimal']['c']

    plt.close()
    fig = plt.figure()
    for i in range(100):
        x_init = np.random.uniform(0.0, 1.0, size=784)
        x_init = np.round(x_init)
        x_sampled = optimal_rbm.generation(x_init, k=1000)
        b = x_sampled.reshape(28,28)
        plt.subplot(10, 10, i+1)
        plt.axis('off')
        plt.imshow(b, cmap='gray')
    
    generate_file = os.path.join(dir_to_save, 'generated_samples.png')
    plt.savefig(generate_file)

if __name__ == "__main__":
    main()

