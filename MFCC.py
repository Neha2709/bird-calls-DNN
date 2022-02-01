# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 11:50:23 2020

@author: Neha Dadarwala
"""


from python_speech_features import mfcc
import scipy.io.wavfile as wav
import os
import numpy


#%%

# directory to put our results in, you can change the name if you like
resultsDirectory =  "testset"

# make a new folder in this directory to save our results in
if not os.path.exists(resultsDirectory):
    os.makedirs(resultsDirectory)

for root, dirs, files in os.walk("test_set", topdown=False):
    for name in dirs:		#name is the name of class(folder)
        resultsDirectory = "testset" + "/" + name
        if not os.path.exists(resultsDirectory):
            os.makedirs(resultsDirectory)
        parts = []		#stores list of all mfcc file name
        parts += [each for each in os.listdir(os.path.join(root,name)) if each.endswith('.wav')]
        print(name, "...")
        for part in parts:
            #data, sampling_rate = librosa.load(os.path.join(root,name,part))
                                            
            #mfccss = librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=39)

            (rate,sig) = wav.read(os.path.join(root,name,part))
            mfccss = mfcc(sig,samplerate=44100,winlen=0.02,winstep=0.01,numcep=39, nfilt=40,nfft=1024,lowfreq=0,highfreq=None,preemph=0.97,ceplifter=22,appendEnergy=True)

            
            # create a file to save our results in
            outputFile = resultsDirectory + "/" + os.path.splitext(part)[0] + ".mfcc"
            file = open(outputFile, 'w+') # make file/over write existing file
            numpy.savetxt(file, mfccss, delimiter=",") #save MFCCs as .csv
            file.close() # close file