
# coding: utf-8

# In[17]:


import os
import sys
import argparse
import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


# In[2]:


'''
Reads the input file and prepares 2 matrices. 
Data matrix of dimensions nX784 
Labels matrix of dimension nX1
'''
def ParseInputData(input_file):
    data = []
    labels = []
    for line in input_file:
        temp_list = line.split(",")
        temp_list = map(lambda x: float(x), temp_list)
        n = len(temp_list)
        data.append(temp_list[:-1])
        labels.append(temp_list[n-1])
    data = np.array(data)
    labels = np.array(labels)
    return data, labels


# In[3]:


def CreateAndFormatBatches(combined_data, batch_size):
    #print "Batch Size = ", batch_size
    #print "len(combined_data) = ", len(combined_data)
    np.random.shuffle(combined_data)
    chunked_data = [combined_data[i:i + batch_size] for i in xrange(0, len(combined_data), batch_size)]
    chunked_data = np.array(chunked_data)
    return chunked_data 

# In[5]:


def CreateOneHotLabels(raw_labels):
    labels = raw_labels.astype(int)
    rows = labels.shape[0]
    cols = 10
    temp = np.zeros((rows,cols))
    temp[np.arange(rows), labels] = 1
    labels = temp
    return labels

# In[4]:


#Neural Network class for two layers
class NeuralNetwork(object):
    def __init__(self, input_dim, layer1_dim, layer2_dim, output_dim):
        
        self.input_dim = input_dim
        self.layer1_dim = layer1_dim
        self.layer2_dim = layer2_dim
        self.output_dim = output_dim
        
        # Initialize weights and biases
        seed = np.sqrt(6/(float)(self.input_dim + self.layer1_dim))
        self.w1 = np.random.uniform(-seed, seed, (self.input_dim, self.layer1_dim))
        self.b1 = np.zeros((1, self.layer1_dim))
        seed = np.sqrt(6/(float)(self.layer1_dim + self.layer2_dim))
        self.w2 = np.random.uniform(-seed, seed, (self.layer1_dim, self.layer2_dim))
        self.b2 = np.zeros((1, self.layer2_dim))
        seed = np.sqrt(6/(float)(self.layer2_dim + self.output_dim))
        self.w3 = np.random.uniform(-seed, seed, (self.layer2_dim, self.output_dim))
        self.b3 = np.zeros((1, self.output_dim))
        self.gamma1 = np.ones((1, self.layer1_dim))
        self.gamma2 = np.ones((1, self.layer2_dim))
        self.beta1 = np.zeros((1, self.layer1_dim))
        self.beta2 = np.zeros((1, self.layer2_dim))
        
        
        # Initialize the intermediate activations to be used in backprop.
        self.a1 = np.array([])
        self.a2 = np.array([])
        self.out = np.array([])
        
        # Keep track of the previous weight gradients to incorporate momentum in the updates
        self.prev_dw1 = np.zeros((self.input_dim, self.layer1_dim))
        self.prev_dw2 = np.zeros((self.layer1_dim, self.layer2_dim))
        self.prev_dw3 = np.zeros((self.layer2_dim, self.output_dim))
        self.prev_db1 = np.zeros((1, self.layer1_dim))
        self.prev_db2 = np.zeros((1, self.layer2_dim))
        self.prev_db3 = np.zeros((1, self.output_dim))
        
        # Keep track of the batch norm forward caches and means for inference
        self.forward_cache_1 = ()
        self.forward_cache_2 = ()
        self.layer1_mean = []
        self.layer1_var = []
        self.layer2_mean = []
        self.layer2_var = []       

    # Forward activation function - computes sigmoid(x)
    def sigmoid(self, x):
        return (1/(float)(1+np.exp(-x)))

    # Backward propogation helper, calculates the sigmoid derivative. Input is sigmoid(x).
    def sigmoid_derivative(self, sig_x):
        return (sig_x * (1-sig_x))

    # Computes the softmax of a vector input
    def softmax(self, scores):
        exp_scores = np.exp(scores)
        prob_vals = exp_scores/np.sum(exp_scores, axis=1, keepdims=True)
        return prob_vals
    
    # Compute the forward pass for batch normalization
    def batchnorm_forward(self, z, gamma, beta, layer, inference=False):
        # Compute the mean and the variance
        eps = 1e-8
        if inference == False:
            mu = np.mean(z, axis=0)
            var = np.var(z, axis=0)
            
            if layer == 1:
                self.layer1_mean.append(mu)
                self.layer1_var.append(var)
            else:
                self.layer2_mean.append(mu)
                self.layer2_var.append(var)
        else:
            if layer == 1:
                mu = np.mean(self.layer1_mean, axis=0)
                var = np.mean(self.layer1_var, axis=0)
            else:
                mu = np.mean(self.layer2_mean, axis=0)
                var = np.mean(self.layer2_var, axis=0)
        
        #Normalize z, adding eps to variance for numerical stability
        z_norm = (z - mu) / np.sqrt(var + eps)
        z_out = gamma * z + beta
        
        to_return = (z, z_norm, mu, var, gamma, beta, eps)
        return z_out, to_return
    
    # Compute batchnorm backpropogation. Equations in the Ioffe & Szegedy paper
    def batchnorm_backward(self, d_prev_layer, bn_forward):
        #Book-keeping values from the batchnorm forward prop
        z, z_norm, mu, var, gamma, beta, eps = bn_forward
        batch_size, dim = z.shape
        z_mu = z - mu
        inv_std = 1./np.sqrt(var + eps)
        
        #Calculate the derivates refering the equations from the paper
        dz_norm = gamma
        d_var = np.sum(dz_norm * z_mu, axis=0) * -0.5 * inv_std**3
        d_mu = np.sum(dz_norm * -inv_std, axis=0) + d_var * np.mean(-2. * z_mu, axis=0)
        dz = (dz_norm * inv_std) + (d_var * 2 * z_mu /(float)(batch_size)) + (d_mu /(float)(batch_size))
        dgamma = np.sum(d_prev_layer * z_norm, axis=0)
        dbeta = np.sum(d_prev_layer, axis=0)
        return dz, dgamma, dbeta
             
    # Function to perform feedforward computations
    def feed_forward(self, input_X, inference=False):
        vfunc = np.vectorize(self.sigmoid, otypes=[np.float])
        z1 = input_X.dot(self.w1) + self.b1
        
        # Batchnorm forward pass - Layer 1
        if inference == False:
            z1_hat, fwd_cache_1 = self.batchnorm_forward(z1, self.gamma1, self.beta1, 1)
        else:
            z1_hat, fwd_cache_1 = self.batchnorm_forward(z1, self.gamma1, self.beta1, 1, True)
            
        self.a1 = vfunc(z1_hat)
        z2 = (self.a1).dot(self.w2) + self.b2
        
        # Batchnorm forward pass - Layer 2
        if inference == False:
            z2_hat, fwd_cache_2 = self.batchnorm_forward(z2, self.gamma2, self.beta2, 2)
        else:
            z2_hat, fwd_cache_2 = self.batchnorm_forward(z2, self.gamma2, self.beta2, 2, True)
            
        self.a2 = vfunc(z2_hat)
        z3 = (self.a2).dot(self.w3) + self.b3
        self.forward_cache_1 = fwd_cache_1
        self.forward_cache_2 = fwd_cache_2
        return self.softmax(z3)

    # Function to perform backward propagation. Returns the loss
    def back_propogation(self, input_X, target_labels, eta, reg_param, momentum):
        '''
        target_labels: True Values in a one hot representation
        eta: Learning Rate
        reg_param: regularization constant
        momentum: momentum for gradient learning
        '''
        # Sigmoid derivative on the first layer activations
        vfunc = np.vectorize(self.sigmoid_derivative, otypes=[np.float])
        num_examples = input_X.shape[0]
        #"Print num examples in backprop: ", num_examples
        beta = self.out - target_labels
        delta3 = beta/num_examples
        sig_der_a2 = vfunc(self.a2)
        temp_2 = beta.dot((self.w3).T)
        delta2 = temp_2 * sig_der_a2
        dz2, dgamma2, dbeta2 = self.batchnorm_backward(delta2, self.forward_cache_2)
        delta2 = delta2 * dz2

        sig_der_a1 = vfunc(self.a1)
        temp_1 = delta2.dot((self.w2).T)
        delta1 = temp_1 * sig_der_a1
        dz1, dgamma1, dbeta1 = self.batchnorm_backward(delta1, self.forward_cache_1)
        delta1 = delta1 * dz1

        dw1 = (input_X.T).dot(delta1) + (momentum * self.prev_dw1)
        dw2 = (self.a1.T).dot(delta2) + (momentum * self.prev_dw2)
        dw3 = (self.a2.T).dot(delta3) + (momentum * self.prev_dw3)
        db1 = np.sum(delta1, axis=0, keepdims=True) + (momentum * self.prev_db1)
        db2 = np.sum(delta2, axis=0, keepdims=True) + (momentum * self.prev_db2)
        db3 = np.sum(delta3, axis=0, keepdims=True) + (momentum * self.prev_db3)

        # Perform the gradient updates
        self.w1 = self.w1 - (eta * dw1) - (2*(reg_param) * self.w1)
        self.w2 = self.w2 - (eta * dw2) - (2*(reg_param) * self.w2)
        self.w3 = self.w3 - (eta * dw3) - (2*(reg_param) * self.w3)
        self.b1 = self.b1 - (eta * db1)
        self.b2 = self.b2 - (eta * db2)
        self.b3 = self.b3 - (eta * db3)
        self.gamma1 = self.gamma1 - (eta * dgamma1)
        self.gamma2 = self.gamma2 - (eta * dgamma2)
        self.beta1 = self.beta1 - (eta * dbeta1)
        self.beta2 = self.beta2 - (eta * dbeta2)

        # Update the prev gradient matrices
        self.prev_dw1 = dw1
        self.prev_dw2 = dw2
        self.prev_dw3 = dw3
        self.prev_db1 = db1
        self.prev_db2 = db2
        self.prev_db3 = db3

        # Calculate the loss for this batch
        loss, correct = self.CalculateLoss(self.out, target_labels, reg_param)
        return loss, correct

    # Calculates the loss with the current values of the weight matrices
    def CalculateLoss(self, pred_probs, target_labels, reg_param):
        correct_labels = np.argmax(target_labels, axis=1)
        pred_labels = np.argmax(pred_probs, axis=1)
        loss = 0.0
        num_correct = 0
        for i,(pred,correct) in enumerate(zip(pred_labels, correct_labels)):
            if pred == correct:
                num_correct += 1
            loss += -np.log(pred_probs[i][correct])
        loss += reg_param * (np.sum(np.square(self.w1)) + np.sum(np.square(self.w2)))
        return loss, num_correct

    '''
    Calculates the Validation or Test loss and returns the loss along with accuracy.
    Performs only feed forward, no backward propagation
    '''
    def CalculateValidTestAccuracy(self, combined_data, reg_param):
        total_loss = 0.0
        num_correct = 0
        #print "combined_data.shape: ", combined_data[0].shape
        total_instances = 0
        #print "Total Instances in valid/test : ", total_instances
        for instance in combined_data:
            loss = 0.0
            total_instances += instance.shape[0]
            input_X = np.array([x[0] for x in instance])
            targets = np.array([x[1] for x in instance])
            #print "input_X shape = ", input_X.shape
            pred_probs = self.feed_forward(input_X, inference=True)
            loss, correct = self.CalculateLoss(pred_probs, targets, reg_param)
            num_correct += correct
            total_loss += loss

        avg_loss = total_loss/(float)(total_instances)
        per_correct = num_correct/(float)(total_instances)
        #print "Total Instances in valid/test : ", total_instances
        #print "Average Loss = ", avg_loss
        #print "Number Correct = ", num_correct
        #print "Total Loss = ", total_loss
        return avg_loss, per_correct

    # Perform Training
    def train(self, train_data, valid_data, test_data, epochs, eta, reg_param, momentum, batch_size):
        # eta: Learning Rate
        # targets are in a one hot representation format for their slice
        model = {}
        losses = []
        valid_losses = []
        test_losses = []
        valid_acc = []
        test_acc = []
        train_acc = []
        non_shuffled_data = train_data
        valid_data = CreateAndFormatBatches(valid_data, batch_size)
        test_data = CreateAndFormatBatches(test_data, batch_size)
        for epoch in range(epochs):
            print " Epoch : ", epoch
            # Random shuffle batches
            train_data = CreateAndFormatBatches(non_shuffled_data, batch_size)
            total_loss = 0.0
            total_instances = train_data.shape[0] * batch_size
            num_correct = 0
            self.layer1_mean = []
            self.layer1_var = []
            self.layer2_mean = []
            self.layer2_var = []
            
            #print "Total Instances : ", total_instances
            for train_instance in train_data:
                #print "train_instance : ", train_instance.shape
    #                 train_instance = np.reshape(train_instance,(self.batch_size,train_instance.shape[0]))
                input_X = np.array([x[0] for x in train_instance])
                targets = np.array([x[1] for x in train_instance])
                self.out = self.feed_forward(input_X)
                loss, correct = self.back_propogation(input_X, targets, eta, reg_param, momentum)
                total_loss += loss
                num_correct += correct
            avg_loss = total_loss/(float)(total_instances)
            cur_acc = num_correct/(float)(total_instances)
            losses.append(avg_loss)
            train_acc.append(cur_acc)
            print "Epoch : ", epoch, ", Avg training loss : ", avg_loss, ", Training Accuracy : ", cur_acc
            valid_loss, valid_accuracy = self.CalculateValidTestAccuracy(valid_data, reg_param)
            valid_losses.append(valid_loss)
            valid_acc.append(valid_accuracy)
            test_loss, test_accuracy = self.CalculateValidTestAccuracy(test_data, reg_param)
            test_losses.append(test_loss)
            test_acc.append(test_accuracy)
            if ((epoch % 10 == 0) or (epoch == 199)):
                print "Epoch : ", epoch, ", Avg validation loss : ", valid_loss, ", Validation accuracy : ", valid_accuracy
                print "Epoch : ", epoch, ", Avg test loss : ", test_loss, ", Test accuracy : ", test_accuracy

        # Create model dictionary
        model['w1'] = self.w1
        model['b1'] = self.b1
        model['w2'] = self.w2
        model['b2'] = self.b2
        model['w3'] = self.w2
        model['b3'] = self.b2
        model['train_loss'] = losses
        model['valid_loss'] = valid_losses
        model['test_loss'] = test_losses
        model['train_acc'] = train_acc
        model['valid_acc'] = valid_acc
        model['test_acc'] = test_acc
        return model


# In[76]:
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, default='models',
                        help = 'Directory to store the models')
    parser.add_argument('--layer1_dim', type=float, default=100.0,
                        help = 'Number of hidden units in first layer')
    parser.add_argument('--layer2_dim', type=float, default=100.0,
                        help = 'Number of hidden units in second layer')
    parser.add_argument('--lr', type=float, default=0.01,
                        help = 'Learning rate used for gradient updates')
    parser.add_argument('--reg_param', type=float, default=0.0001,
                        help = 'Regularization Parameter for weight updates')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help = 'Momentum used for incremental parameter updates')
    parser.add_argument('--batch_size', type=float, default=32.0,
                        help = 'Batch Size')
    parser.add_argument('--epochs', type=float, default=200.0,
                        help = 'Number of iterations to be run for the dataset')
    parser.add_argument('--train', type=str, default='data/digitstrain.txt',
                        help = 'Training data file')
    parser.add_argument('--valid', type=str, default='data/digitsvalid.txt',
                        help = 'Validation data file')
    parser.add_argument('--test', type=str, default='data/digitstest.txt',
                        help = 'Test data file')
    args = parser.parse_args()

    # Hard Coded Values
    in_dim = 784
    out_dim = 10

    print(sys.argv)
    # Read the train, valid and test data
    f = open(args.train, 'rb')
    data, labels = ParseInputData(f)
    train_labels = CreateOneHotLabels(labels)
    f.close()
    v = open(args.valid, 'rb')
    valid_data, valid_labels = ParseInputData(v)
    valid_labels = CreateOneHotLabels(valid_labels)
    v.close()
    t = open(args.test, 'rb')
    test_data, test_labels = ParseInputData(t)
    test_labels = CreateOneHotLabels(test_labels)
    t.close()
    train = np.array(zip(data, train_labels))
    valid = np.array(zip(valid_data, valid_labels))
    test = np.array(zip(test_data, test_labels))

    np.random.seed(seed=10)
    nn = NeuralNetwork(input_dim=in_dim, layer1_dim=int(args.layer1_dim), layer2_dim=int(args.layer2_dim), output_dim=out_dim)
    model = nn.train(train_data=train, valid_data=valid, test_data=test, epochs=int(args.epochs), eta=args.lr, reg_param=args.reg_param, 
              momentum=args.momentum, batch_size=32)

    subdir = 'Layer_2_BN_Hidden_' + str(args.layer1_dim) + '_' + str(args.layer2_dim)
    filename = 'model_lr' + str(args.lr) + '_reg' + str(args.reg_param) + '_mom' + str(args.momentum) + \
                'batch' + str(args.batch_size) + 'valid_max_' + str(model['max_valid_acc']) + '.pickle'
    dir_to_save = os.path.join(args.out_dir, subdir)
    
    try:
        os.makedirs(dir_to_save)
    except:
        pass

    pickle_file = os.path.join(dir_to_save, filename)
    print "Saving file = ", pickle_file
    with open(pickle_file, 'wb') as p_pickle:
        pickle.dump(model, p_pickle)

if __name__ == "__main__":
    main()




