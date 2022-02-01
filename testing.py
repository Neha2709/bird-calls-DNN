# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 23:00:33 2020

@author: Neha Dadarwala
"""

from python_speech_features import mfcc
import scipy.io.wavfile as wav
import numpy
import pickle
import numpy as np
import tensorflow as tf

#%%

#(rate,sig) = wav.read('../25 species/test_set/Greater Coucal/part (38).wav')
(rate,sig) = wav.read('part_17.wav')
mfccss = mfcc(sig,samplerate=44100,winlen=0.02,winstep=0.01,numcep=39, nfilt=40,nfft=1024,lowfreq=0,highfreq=None,preemph=0.97,ceplifter=22,appendEnergy=True)


# create a file to save our results in
outputFile = "test.mfcc"
file = open(outputFile, 'w+') # make file/over write existing file
numpy.savetxt(file, mfccss, delimiter=" ") #save MFCCs as .csv
file.close() # close file

#%%

# Create model
def multilayer_perceptron(_X, _weights, _biases):
    # Hidden layer with RELU activation
    layer_1 = tf.nn.relu(tf.add(tf.matmul(_X, _weights['h1']), _biases['b1']))
    # Hidden layer with sigmoid activation
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, _weights['h2']), _biases['b2']))
    layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2, _weights['h3']), _biases['b3']))
    #layer_4 = tf.nn.relu(tf.add(tf.matmul(layer_3, _weights['h4']), _biases['b4']))
    # layer_5 = tf.nn.sigmoid(tf.add(tf.matmul(layer_4, _weights['h5']), _biases['b5']))
    return tf.nn.softmax(tf.matmul(layer_3, _weights['out']) + _biases['out'])



def indices(a, func):
    """Finds elements of the matrix that correspond to a function"""
    return [i for (i, val) in enumerate(a) if func(val)]

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def load():
    
    n_classes = 56 # Number of classes in bird data
    parametersFileDir = "trainset/parameters_mfcc_3.pkl"
    
    n_input = 585  # input dimensionality

    x = tf.placeholder("float", [None, n_input])
    y = tf.placeholder("float", [None, n_classes])
    
    #print("Loading saved Weights ...")
    file_ID = parametersFileDir
    f = open(file_ID, "rb")
    W = pickle.load(f)
    b = pickle.load(f)

    # print "b1 = ", b['b1']
    # print "b2 = ", b['b2']
    # print "b3 = ", b['out']

    weights = {
        'h1': tf.Variable(W['h1']),
        'h2': tf.Variable(W['h2']),
        'h3': tf.Variable(W['h3']),
        #'h4': tf.Variable(W['h4']),
        # 'h5': tf.Variable(W['h5']),
        'out': tf.Variable(W['out'])
    }

    biases = {
        'b1': tf.Variable(b['b1']),
        'b2': tf.Variable(b['b2']),
        'b3': tf.Variable(b['b3']),
        #'b4': tf.Variable(b['b4']),
        # 'b5': tf.Variable(b['b5']),
        'out': tf.Variable(b['out'])
    }
    # print type(b['b1'])
    # print type(biases['b1'])

    f.close()


    pred = multilayer_perceptron(x, weights, biases)

    #print("Testing the Neural Network")
    init = tf.initialize_all_variables()
    
    
    with tf.Session() as sess:
        sess.run(init)
        file_specified = 'test.mfcc'
        example = np.loadtxt(file_specified)
        i = 0
        rows, cols = example.shape
        context = np.zeros((rows - 14, 15 * cols))  # 15 contextual frames
        while i <= (rows - 15):
            ex = example[i:i + 15, :].ravel()
            ex = np.reshape(ex, (1, ex.shape[0]))
            context[i:i + 1, :] = ex
            i += 1
        # see = tf.argmax(pred, 1)
        see = tf.reduce_sum(pred, 0)

        confidence_matrix = softmax(see.eval({x: context}))
        
        #print(confidence_matrix)
        
        confidence_matrix = confidence_matrix * 100
       

        list1 = ['Ashy Prinia','Barn owl', 'Baya Weaver', 'Black Drongo','Black Kite' , 'blackandyellow grosbeak ',
                 'blackcrested tit','black throated tit ', 'chestnutcrowned laughingthrush ','Common Myna ',
                 'Common Tailorbird', 'Coppersmith Barbet', 'darksided flycatcher ', 'eurasian treecreeper ', 
                 'golden bushrobin ', 'Greater Coucal', 'great barbet ', 'Green Bee Eater', 'greyheaded canary flycatcher ',
                 'greyhooded warbler ', 'greywinged blackbird ', 'grey bellied cuckoo ', 'grey bushchat ',  'himalayan monal ',
                 'House Crow', 'House Sparrow', 'humes warbler ', 'Indian Peafowl', 'Indian Robin', 'Jungle Babbler',
                 'largebilled crow ', 'large hawkcuckoo ', 'Laughing Dove', 'lesser cuckoo', 'Little Egret',
                 'orangeflanked bushrobin ', 'oriental cuckoo ', 'palerumped warbler ', 'Purple Sunbird' ,
                 'Red vented bulbul', 'Red Whiskered bulbul' , 'redbilled chough ', 'Rock Dove', 'rock bunting ',
                 'Rose ringed parrot', 'rufousbellied niltava ',' rufous gorgetted flycatcher ', 'spotted nutcracker ',
                 'streaked laughingthrush ', 'variegated laughingthrush ' , 'western tragopan ', 'whistlers warbler ',
                 'White Throated Kingfisher', 'whitebrowed fulvetta ' , 'whitecheeked nuthatch ',  'yellowbellied fantail ']
                 
                 
        #Top Three Labels
        res = np.asarray(confidence_matrix)

        result = {}
        
        for i in range(3):
            rank = {}
            product = np.argmax(res)
            #if (round(res[product],2) == 100.0):
                #result = {"Error":"Bird Not Detected"}
                #break
            tmp = str(round(res[product],5))
            rank[list1[product]] = tmp
            result[i] = rank
            res[product] = 0
            
        #print(result)
        print(result)
     
           

load()
