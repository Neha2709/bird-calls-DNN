# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 13:38:00 2020

@author: Neha Dadarwala
"""


#!/usr/local/bin/python

import numpy as np
import tensorflow as tf
import os
import pickle

################    All Constants and paths used    #####################
path = ""
n_classes = 56 # Number of classes in bird data
parametersFileDir = "trainset/parameters_mfcc_3.pkl"
relativePathForTest = "testset"
testFilesExtension = '.mfcc'
confMatFileDirectory = 'trainset/confmat.txt'


def indices(a, func):
    """Finds elements of the matrix that correspond to a function"""
    return [i for (i, val) in enumerate(a) if func(val)]


#test_labels_dense = np.loadtxt('./data/ground_truth.txt');
#test_labels_dense = test_labels_dense.astype(int)
# test_y = dense_to_one_hot(test_labels_dense, num_classes = n_classes)
# print train_labels_dense
# plot_data(train_X, train_labels_dense)
# time.sleep(10)
# plt.close('all')
print("Data Loaded and processed ...")
################## Neural Networks Training #################################

print("Verifying Neural Network Parameters ...")

# Network Parameters
n_hidden_1 = 512 # 1st layer num features
n_hidden_2 = 512 # 2nd layer num features
n_hidden_3 = 512 # 3rd layer num features
n_hidden_4 = 512
n_hidden_5 = 512
n_input = 585 # input dimensionality

x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

# Create model
def multilayer_perceptron(_X, _weights, _biases):
    #Hidden layer with RELU activation
    layer_1 = tf.nn.relu(tf.add(tf.matmul(_X, _weights['h1']), _biases['b1']))
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, _weights['h2']), _biases['b2']))
    layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2, _weights['h3']), _biases['b3']))
    #layer_4 = tf.nn.relu(tf.add(tf.matmul(layer_3, _weights['h4']), _biases['b4']))
    #layer_5 = tf.nn.relu(tf.add(tf.matmul(layer_4, _weights['h5']), _biases['b5']))
    return tf.nn.softmax(tf.matmul(layer_3, _weights['out']) + _biases['out'])

print("Loading saved Weights ...")
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
    #'h5': tf.Variable(W['h5']),
    'out': tf.Variable(W['out'])
    }

biases = {
    'b1': tf.Variable(b['b1']),
    'b2': tf.Variable(b['b2']),
    'b3': tf.Variable(b['b3']),
    #'b4': tf.Variable(b['b4']),
    #'b5': tf.Variable(b['b5']),
    'out': tf.Variable(b['out'])
}
# print type(b['b1'])
# print type(biases['b1'])

f.close()

# layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
# layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))


pred = multilayer_perceptron(x, weights, biases)

print("Testing the Neural Network")
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    num_examples = 0
    for root, dirs, files in os.walk(relativePathForTest, topdown=False):
        for name in dirs:
            parts = []
            parts += [each for each in os.listdir(os.path.join(root,name)) if each.endswith(testFilesExtension)]

            for part in parts:
                num_examples += 1

    # Test model

    # likelihood = tf.argmax(tf.reduce_mean(pred, 0),1)
    test_labels_dense = np.zeros(num_examples)
    test_labels_dense = test_labels_dense.astype(int)
    label = np.zeros(test_labels_dense.shape[0])
    ind = 0
    gt = 0

#if len(sys.argv) == 1:
    for root, dirs, files in os.walk(relativePathForTest, topdown=False):
        for name in dirs:
            print(name)
            parts = []
            parts += [each for each in os.listdir(os.path.join(root,name)) if each.endswith(testFilesExtension)]

            for part in parts:

                example = np.loadtxt(os.path.join(root,name,part),delimiter=',')
                i = 0
                rows, cols = example.shape
                context = np.zeros((rows-14,15*cols)) # 15 contextual frames
                while i <= (rows - 15):
                    ex = example[i:i+15,:].ravel()
                    ex = np.reshape(ex,(1,ex.shape[0]))
                    context[i:i+1,:] = ex
                    i += 1
                # see = tf.argmax(pred, 1)
                see = tf.reduce_sum(pred,0)
                matrix = np.asarray(see.eval({x: context}))
                product = np.argmax(matrix)
                #matrix[product] = 0
                #product = np.argmax(matrix)
                #matrix[product] = 0
                #product = np.argmax(matrix)
                
                # product = pred.eval({x: example})
                # sums = np.sum(product,1, keepdims = True)
                # sumtodiv = np.tile(sums,26)
                # prob = np.divide(product,sumtodiv)
                # prodofprob = np.cumsum(np.log(prob),0)

                # product = product.reshape((product.shape[0],1))
                # print product.shape
                # np.savetxt('./data/product.txt', product, delimiter = ' ')
                # if i == 0:
                # 	 np.savetxt('output.txt',np.asarray(pred.eval({x: example})), delimiter = ' ');
                # label[ind] = product
                # print mode(product)
                # label[ind],_ = mode(product,axis=None)
                label[ind] = product
                test_labels_dense[ind] = gt
                # print label.shape
                ind += 1
            gt += 1
            

label = label.astype(int)
conf = np.zeros((n_classes, n_classes), dtype=np.int32)
for i in range(label.shape[0]):
    conf[test_labels_dense[i], label[i]] += 1

#print(conf)
np.savetxt(confMatFileDirectory, conf, fmt='%i', delimiter=' ')
accuracy = np.sum(np.diag(conf))
accuracy = (float(accuracy) / label.shape[0]) * 100
print("Accuracy is %.4f " % accuracy)
#plt.close('all')
#plt.savefig(pp, format='pdf')
#pp.close()
#plt.show()

#%%

import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline
from sklearn.metrics import confusion_matrix
matrix = confusion_matrix(test_labels_dense, label)
sns.heatmap(matrix,annot=True,cbar=False)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix')

#%%

from sklearn.metrics import classification_report
print(classification_report(test_labels_dense,label))