# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 11:50:26 2020

@author: Neha Dadarwala
"""


import numpy as np
import os

#%%

cl = 0  # number of classes
sum = 0  # number of training samples total

f_handle = open('trainset/train_mel48.dat', 'ab')  # open file for append in binary format write only
for root, dirs, files in os.walk("trainset", topdown=False):
    for name in dirs:  # name is the name of class(folder)
        parts = []  # stores list of all mfcc file name
        parts += [each for each in os.listdir(os.path.join(root, name)) if each.endswith('.mfcc')]
        print(name, "...")
        for part in parts:  # for each mfcc file in parts
            example = np.loadtxt(os.path.join(root, name, part), delimiter=',')
            i = 0
            rows = example.shape[0]
            while i <= (rows - 15):
                context = example[i:i + 15, :].ravel()  # 15 mfcc in 1D
                ex = np.append(context, cl)  # appending class number to mfcc
                ex = np.reshape(ex, (1, ex.shape[0]))
                np.savetxt(f_handle, ex)  # write ex in train_mel.dat fiile
                sum += 1
                # print ex.shape
                i += 1
            print("No. of context windows: %d" % i)
        cl += 1
print("No. of training examples: %d" % sum)

f_handle.close()

#%%

A = np.loadtxt('trainset/train_mel48.dat')
np.random.shuffle(A)
np.savetxt('trainset/train_melfilter48.dat',A,delimiter = ' ')