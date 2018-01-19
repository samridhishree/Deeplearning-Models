
# coding: utf-8

# In[36]:

from __future__ import print_function
import os
import sys
import argparse
import operator
import itertools
import time
from collections import defaultdict, OrderedDict
import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt


# In[37]:


wids = defaultdict(lambda: len(wids))
id2word = {} 
word_freq = defaultdict(int)
ngram_freq = defaultdict(int)
START = "START"
END = "END"
UNK = "UNK"


# Prepare the vocabulary according to the truncated threshold
def PrepareVocab(filename, N, threshold):
    global wids
    global word_freq
    global id2word
    f = open(filename, 'rb')
    
    # Populate the word frequency dictionary
    for line in f:
        line = line.lower()
        line = START + " " + line + " " + END
        line = line.strip().split(' ')
        for word in line:
            word_freq[word] += 1
        # Create ngrams and update the ngram frequencies
        ngrams = [line[i:i+N] for i in xrange(len(line)-N+1)]
        for ngram in ngrams:
            ngram_key = ' '.join(ngram)
            ngram_freq[ngram_key] += 1
    TruncateVocab(threshold)
    id2word = {i:w for w,i in wids.iteritems()}

# Creates indexed ngrams for training and validation set while taking care of UNK
def PrepareData(filename, N):
    data = []
    labels = []
    f = open(filename, 'rb')
    for line in f:
        ngrams = []
        tagged_sentence = []
        line = line.lower()
        line = START + " " + line + " " + END
        line = line.strip().split(' ')
        tagged_sentence = [WordToInt(word) for word in line]
        ngrams = [tagged_sentence[i:i+N] for i in xrange(len(tagged_sentence)-N+1)]
        for ngram in ngrams:
            data.append(ngram[:-1])
            labels.append(ngram[N-1])
    data = np.array(data)
    labels = np.array(labels)
    return data, labels  

def TruncateVocab(threshold):
    global wids
    global word_freq
    sorted_freq = sorted(word_freq.items(), key=operator.itemgetter(1), reverse=True)
    sorted_freq = sorted_freq[0:threshold]
    for key,val in sorted_freq:
        wids[key]


# In[38]:

def WordToInt(word):
    if word in wids:
        return wids[word]
    else:
        return wids[UNK]

def IntToWord(word_id):
    if word_id in id2word:
        return id2word[word_id]
    else:
        return UNK


# In[39]:


# Save the top N ngrams to the given file
def TopNgrams(N, filename):
    w = open(filename, 'wb')
    sorted_ngram = sorted(ngram_freq.items(), key=operator.itemgetter(1), reverse=True)
    sorted_ngram = sorted_ngram[0:N]
    for (ngram,count) in sorted_ngram:
        w.write(ngram + "\n")
    w.close()


# In[40]:


# Create batches of batch_size
def CreateBatches(combined_data, batch_size):
    np.random.shuffle(combined_data)
    chunked_data = [combined_data[i:i + batch_size] for i in xrange(0, len(combined_data), batch_size)]
    chunked_data = np.array(chunked_data)
    return chunked_data

# Create one-hot representation of the target word
def CreateOneHotLabels(raw_labels):
    global wids
    labels = raw_labels.astype(int)
    rows = labels.shape[0]
    cols = len(wids)
    temp = np.zeros((rows,cols))
    for row in range(rows):
        temp[row, labels[row]] = 1
    labels = temp
    return labels


# In[48]:


#Neural Network class
class NeuralNetwork(object):
    def __init__(self, input_dim, hidden_dim, output_dim, n, log_file):
        '''
        input_dim: Dimensionality of input data - embedding size (D)
        hidden_dim: No. of neurons in the hidden layer - hidden size (H)
        output_dim: Output Dimension - Vocab Size (V)
        n - ngrams 'n'
        '''
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n = n
        self.f_out = log_file
        
        # Initialize the weights and biases.
        seed = np.sqrt(6/(float)(self.input_dim + self.hidden_dim))
        self.w1 = np.random.uniform(-seed, seed, (((n-1) * self.input_dim), self.hidden_dim))
        self.b1 = np.zeros((1, self.hidden_dim))
        seed = np.sqrt(6/(float)(self.hidden_dim + self.output_dim))
        self.w2 = np.random.uniform(-seed, seed, (self.hidden_dim, self.output_dim))
        self.b2 = np.zeros((1, self.output_dim))
        self.cw = np.random.normal(0, 0.1, (self.output_dim, self.input_dim))
        
        # Initialize the intermediate activations to be used in backprop. Set to 1
        self.a1 = np.array([])
        self.out = np.array([])
    
    # Computes the softmax of a vector input
    def softmax(self, scores):
        exp_scores = np.exp(scores)
        prob_vals = exp_scores/np.sum(exp_scores, axis=1, keepdims=True)
        return prob_vals

    def get_embeddings(self, input_X):
        b = input_X.shape[0]
        n = input_X.shape[1]
        embeddings = self.cw[input_X[range(b), :]]
        embeddings = np.reshape(embeddings, (b, (n * self.cw.shape[1])))
        return embeddings
    
    # Function to perform feedforward computations
    def feed_forward(self, embed_x):
        self.a1 = embed_x.dot(self.w1) + self.b1
        z2 = (self.a1).dot(self.w2) + self.b2
        return self.softmax(z2)
    
     # Function to perform backward propagation. Returns the loss
    def back_propogation(self, embed_x, input_X, target_labels, eta):
        '''
        target_labels: True Values in a one hot representation
        eta: Learning Rate
        '''
        num_examples = input_X.shape[0]
        beta = self.out - target_labels
        delta1 = beta/num_examples
        delta2 = delta1.dot((self.w2).T)
        dw1 = (embed_x.T).dot(delta2)
        dw2 = (self.a1.T).dot(delta1)
        db1 = np.sum(delta2, axis=0, keepdims=True)
        db2 = np.sum(delta1, axis=0, keepdims=True)
        dx = (delta2.dot(self.w1.T)) * num_examples
        self.w1 = self.w1 - (eta * dw1)
        self.w2 = self.w2 - (eta * dw2)
        self.b1 = self.b1 - (eta * db1)
        self.b2 = self.b2 - (eta * db2)
        
        # Update the embedding matrix cx
        D = self.input_dim
        self.cw[input_X[:, 0]] = self.cw[input_X[:, 0]] - (eta * dx[:, 0:D])
        self.cw[input_X[:, 1]] = self.cw[input_X[:, 1]] - (eta * dx[:, D:(2*D)])
        self.cw[input_X[:, 2]] = self.cw[input_X[:, 2]] - (eta * dx[:, (2*D):(3*D)])
        
        # Calculate the loss for this batch
        loss, correct = self.get_loss(self.out, target_labels)
        return loss, correct
    
    # Calculates the loss with the current values of the weight matrices
    def get_loss(self, pred_probs, target_labels):
        correct_labels = np.argmax(target_labels, axis=1)
        pred_labels = np.argmax(pred_probs, axis=1)
        num_examples = pred_probs.shape[0]
        loss = 0.0
        num_correct = 0
        for i,(pred,correct) in enumerate(zip(pred_labels, correct_labels)):
            if pred == correct:
                num_correct += 1
            loss += -np.log(pred_probs[i][correct])
        return loss, num_correct
    
    '''
    Calculates the Validation Loss and Perplaxity.
    Performs only feed forward, no backward propagation
    '''
    def get_valid_ppl(self, combined_data):
        total_loss = 0.0
        num_correct = 0
        total_instances = 0
        for instance in combined_data:
            loss = 0.0
            total_instances += instance.shape[0]
            input_X = np.array([x[0] for x in instance])
            targets = np.array([x[1] for x in instance])
            embed_x = self.get_embeddings(input_X)
            pred_probs = self.feed_forward(embed_x)
            loss, correct = self.get_loss(pred_probs, targets)
            num_correct += correct
            total_loss += loss     
        avg_loss = total_loss/(float)(total_instances)
        per_correct = num_correct/(float)(total_instances)
        ppl = np.exp(avg_loss)
        return avg_loss, per_correct, ppl
    
    # Perform Training
    def train(self, train_data, valid_data, epochs, eta, batch_size):
        # eta: Learning Rate
        # targets are in a one hot representation format for their slice
        model = {}
        optimal = {}
        losses = []
        valid_losses = []
        valid_acc = []
        valid_ppl = []
        train_acc = []
        train_ppl = []
        min_val_ppl = np.inf
        non_shuffled_data = train_data
        valid_data = CreateBatches(valid_data, batch_size)
        for epoch in range(epochs):
            self.f_out.write(" Epoch : %d\n" % epoch)
            start = time.time()
            # Random shuffle batches
            train_data = CreateBatches(non_shuffled_data, batch_size)
            total_loss = 0.0
            total_instances = train_data.shape[0] * batch_size
            num_correct = 0
            for train_instance in train_data:
                input_X = np.array([x[0] for x in train_instance])
                targets = np.array([x[1] for x in train_instance])
                embed_x = self.get_embeddings(input_X)
                self.out = self.feed_forward(embed_x)
                loss, correct = self.back_propogation(embed_x, input_X, targets, eta)
                total_loss += loss
                num_correct += correct
            avg_loss = total_loss/(float)(total_instances)
            cur_acc = num_correct/(float)(total_instances)
            cur_ppl = np.exp(avg_loss)
            losses.append(avg_loss)
            train_acc.append(cur_acc)
            train_ppl.append(cur_ppl)
            self.f_out.write("Epoch : %d , Avg training loss : %f , Training Perplexity : %f \n" % (epoch, avg_loss, cur_ppl))
            valid_loss, valid_accuracy, v_ppl = self.get_valid_ppl(valid_data)
            valid_losses.append(valid_loss)
            valid_acc.append(valid_accuracy)
            valid_ppl.append(v_ppl)
            end = time.time()
            self.f_out.write("Time taken = %f\n" % (end-start))
            if ((epoch % 10 == 0) or (epoch == (epochs - 1))):
                self.f_out.write("Epoch : %d , Avg validation loss : %f , Validation Perplexity : %f\n" % (epoch, valid_loss, v_ppl))

            # Update the optimal model
            if v_ppl < min_val_ppl:
                min_val_ppl = v_ppl
                optimal['w1'] = self.w1
                optimal['b1'] = self.b1
                optimal['w2'] = self.w2
                optimal['b2'] = self.b2
                optimal['cw'] = self.cw
                optimal['vald_ppl'] = v_ppl
                optimal['val_loss'] = valid_loss

        min_valid = np.argmin(valid_ppl)
        self.f_out.write("Maximum validation accuracy at epoch : %f with a value of %f\n" % (min_valid , np.min(valid_ppl)))
        model['w1'] = self.w1
        model['b1'] = self.b1
        model['w2'] = self.w2
        model['b2'] = self.b2
        model['train_loss'] = losses
        model['valid_loss'] = valid_losses
        model['train_acc'] = train_acc
        model['valid_acc'] = valid_acc
        model['train_ppl'] = train_ppl
        model['valid_acc'] = valid_ppl
        model['min_valid_ppl'] = np.min(valid_ppl)
        model['optimal'] = optimal
        return model

# In[ ]:

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, default='models',
                        help = 'Directory to store the models')
    parser.add_argument('--input_dim', type=int, default=16,
                        help = 'Embedding Dimension')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help = 'Number of hidden units')
    parser.add_argument('--lr', type=float, default=0.01,
                        help = 'Learning rate used for gradient updates')
    parser.add_argument('--batch_size', type=int, default=512,
                        help = 'Batch Size')
    parser.add_argument('--epochs', type=int, default=100,
                        help = 'Number of iterations to be run for the dataset')
    parser.add_argument('--train', type=str, default='data/train.txt',
                        help = 'Training data file')
    parser.add_argument('--valid', type=str, default='data/val.txt',
                        help = 'Validation data file')
    parser.add_argument('--log_file', type=str, default='logs/run_ouput.log',
                        help = 'File to log the model runs')


    args = parser.parse_args()

    PrepareVocab(args.train, 4, 8000)
    train_data, train_labels = PrepareData(args.train, 4)
    train_labels = CreateOneHotLabels(train_labels)
    train = np.array(zip(train_data, train_labels))

    valid_data, valid_labels = PrepareData(args.valid, 4)
    valid_labels = CreateOneHotLabels(valid_labels)
    valid = np.array(zip(valid_data, valid_labels))

    vocab_size = len(wids)
    f_out = open(args.log_file, 'w', 0)
    print (sys.argv)

    # Create a neural network obejct and train
    nn = NeuralNetwork(input_dim=args.input_dim, hidden_dim=args.hidden_dim, output_dim=vocab_size, n=4, log_file=f_out)
    model = nn.train(train_data=train, valid_data=valid, epochs=args.epochs, eta=args.lr, batch_size=args.batch_size)
    f_out.close()

    subdir = 'Linear_Hidden_' + str(args.hidden_dim)
    filename = 'model_lr' + str(args.lr) + 'batch' + str(args.batch_size) + 'valid_min_' + str(model['min_valid_ppl']) + '.pickle'
    dir_to_save = os.path.join(args.out_dir, subdir)
    
    try:
        os.makedirs(dir_to_save)
    except:
        pass

    pickle_file = os.path.join(dir_to_save, filename)
    print ("Saving file = %s" % pickle_file)
    with open(pickle_file, 'wb') as p_pickle:
        pickle.dump(model, p_pickle)

if __name__ == "__main__":
    main()

