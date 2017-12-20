# coding: utf-8

import os
import sys
import operator
import itertools
import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt


input_file = sys.argv[1]
model_pickle = sys.argv[2]
vocab_pickle = sys.argv[3]
output_file = sys.argv[4]
word_output_file = sys.argv[5]

model = pickle.load(open(model_pickle, 'r'))
optimal = model['optimal']
wids = pickle.load(open(vocab_pickle, 'r'))
id2word = {i:w for w,i in wids.iteritems()}
START = "START"
END = "END"
UNK = "UNK"

# Read the required parameters
w1 = optimal['w1']
b1 = optimal['b1']
w2 = optimal['w2']
b2 = optimal['b2']
cw = optimal['cw']

def IntToWord(word_id):
    if word_id in id2word:
        return id2word[word_id]
    else:
        return UNK

def WordToInt(word):
    if word in wids:
        return wids[word]
    else:
        return wids[UNK]

def get_embeddings(input_X):
    b = input_X.shape[0]
    n = input_X.shape[1]
    embeddings = cw[input_X[range(b), :]]
    embeddings = np.reshape(embeddings, (b, (n * cw.shape[1])))
    return embeddings

def softmax(scores):
    exp_scores = np.exp(scores)
    prob_vals = exp_scores/np.sum(exp_scores, axis=1, keepdims=True)
    return prob_vals
    
# Function to perform feedforward computations
def generate_next_word(embed_x):
    a1 = embed_x.dot(w1) + b1
    z2 = a1.dot(w2) + b2
    out = softmax(z2)
    #print "out.shape = ", out.shape
    pred_wid = np.argmax(out, axis=1)[0]
    #print "pred_wid = ", pred_wid
    return IntToWord(pred_wid)

def compute_euc_distance(w1, w2):
    dist = np.linalg.norm(w1-w2)

def generate():
    f = open(input_file, 'rb')
    w = open(output_file, 'w', 0)
    for line in f:
        print "Generating for line = ", line
        ngrams = []
        tagged_sentence = []
        line = line.lower()
        line = line.strip().split(' ')
        tagged_sentence = [WordToInt(word) for word in line]
        ngram = np.array(tagged_sentence)
        ngram = np.reshape(ngram, (1, ngram.shape[0]))
        embed_x = get_embeddings(ngram)
        generated_sentence = line
        num_words = len(generated_sentence)
        start = 0
        while num_words <= 10: #or generated_sentence[-1] == END:
            next_word = generate_next_word(embed_x)
            generated_sentence.append(next_word)
            num_words = len(generated_sentence)
            start += 1
            ngram = [WordToInt(word) for word in generated_sentence[start:]]
            ngram = np.array(ngram)
            ngram = np.reshape(ngram, (1, ngram.shape[0]))
            embed_x = get_embeddings(ngram)
        gen = ' '.join(generated_sentence) + '\n'
        print "Sentence generated = ", gen
        w.write(gen)
        #w.write('\n')
    w.close()
    f.close()

def find_similar_words(target_words):
    writer = open(word_output_file, 'w')
    for word in target_words:
        print "Finding for word = ", word
        wid = WordToInt(word)
        embedding = cw[wid]
        similar = {}
        for w_n in wids:
            n_id = WordToInt(w_n)
            n_emb = cw[n_id]
            dist = np.linalg.norm(embedding-n_emb)
            similar[w_n] = dist
        sorted_similar = sorted(similar.items(), key=operator.itemgetter(1))
        similar = sorted_similar[0:30]
        #print similar
        sorted_similar = [w for (w,dis) in similar]
        to_write = word + ' : ' + str(sorted_similar) + '\n\n'
        writer.write(to_write)
    writer.close()

def generate_scatter_plot(x, y, words):
    colors = np.random.rand(len(x))
    fig, ax = plt.subplots()
    ax.scatter(x, y, c=colors)
    for i, txt in enumerate(words):
        ax.annotate(txt, (x[i],y[i]))
    plt.savefig('word_scatter.png')

find_similar_words(['president', 'the', 'company', 'government', 'city', 'chief', 'stock', 'money', 'you', 'business', 'america', 'question', 'leader', 'subsidiary', 'officer'])
#generate()










