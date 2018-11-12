#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 09:57:41 2018

@author: Renee
"""


import struct
import numpy as np
import os
import sys

######## DATA MANIPULATION

def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)

def get_file(data_folder,filename):
#    script_dir = os.path.dirname(__file__)
    # rel_path = data_folder+filename
    abs_file_path = os.path.join(data_folder,filename)
    data = read_idx(abs_file_path)
    # rows = len(data)
    return data

def get_feature1(data):
    # Scale each data:divide each value by 255 and round it).
    data0 = data/float(255)
    n = len(data0)

    # Make the 2D vector a 1D vector of size 784 to use in the Perceptrons
    data_f1 = np.reshape(np.ravel(data0),(n,784))

    # Take the bias term as another feature and pass the input value 1 for that feature.
#    bias = np.ones((len(data_1d),1))
#    out = np.hstack([data_1d,bias])

    return data_f1

def get_feature2(data):

    # Max-pooling using sliding window

    # Scale each data:divide each value by 255 and round it).
    data0 = data/float(255)
    n = len(data0)

    # Make the 2D vector a 1D vector of size 784 to use in the Perceptrons
    data = np.reshape(np.ravel(data0),(n,784))
    data = data.copy()
    data_f2[0][0] = data_f2[0:1][0:1]

    # Take the bias term as another feature and pass the input value 1 for that feature.
#    bias = np.ones((len(data_1d),1))
#    out = np.hstack([data_1d,bias])

    return data_f2

# Change y into booleans
def label_bool(digit, y):
    temp = np.where(y == digit, 1,0)
    return temp



# Calculate probability
def probability(x, weight):
    z = np.dot(x, weight)
    prob = 1/(1+np.exp(-z))
    # pred = np.rint(prob)
    return prob


######## ESTIMATE WEIGHTS

# Update weight using batch gradient descent
def gd_weight(x, y, l_rate, reg,lam):
    np.random.seed(123)
    d = x.shape[1]
    weight = np.random.randn(d)
    # w_mat = np.empty((0,d),float)
    w_delta = np.zeros((d,1))
    prob = probability(x,weight)
    w_delta = np.dot(x.T, prob - y) - lam*weight*reg
    print(w_delta)
    weight += l_rate * w_delta
    # w_mat = np.vstack(w_mat,weight)
    return weight

# Update weight using stochastic gradient descent
def sgd_weight(x, y, l_rate, reg,lam):
    np.random.seed(123)
    n = x.shape[0]
    d = x.shape[1]
    weight = np.random.randn(d)
    w_mat = np.empty((0,d),float)
    for i in range(n):
        prob = probability(x[i],weight)
        w_delta = (prob - y[i]) * x[i] - lam*weight*reg
        weight += l_rate * w_delta
        w_mat = np.vstack((w_mat,weight))
    return w_mat

# Make a prediction using logistic regression with training weights
def predict(x,weight):
    for digit in range(10):

        pred = np.argmax(np.)


# Calculating accuracy for digit
def evaluate(pred,y):
    y = y.flatten()
    pred = pred.flatten()
    N1 = y.shape[0]
    N2 = pred.shape[0]
    assert (N1==N2)
    accuracy = 0
    F1 = 0
    for i in range(10):
        TP = np.sum(np.logical_and(pred==i, y==i))
        TN = np.sum(np.logical_and(pred!=i, y!=i))
        FN = np.sum(np.logical_and(pred!=i, y==i))
        FP = np.sum(np.logical_and(pred==i, y!=i))
        assert(TP + TN + FN + FP == N1)
        print (TP,TN,FN,FP)     #debug use
        acc = float(TP+TN)/N1
        accuracy+=acc
#        if TP+FP == 0:
#            print("precision == 0") #debug use
#            precision = 0
#        else:
#            precision = float(TP)/(TP+FP)
#        if TP+FN == 0:
#            print("precision == 0")   #debug use
#            recall = 0
#        else:
#            recall = float(TP)/(TP+FN)
#        if precision + recall == 0:
#            print("precision == 0")    #debug use
#            F1_score = 0
#        else:
#            F1_score = 2.0 * precision * recall / (precision + recall)
#        F1+=F1_score
    print ("Accuracy ", accuracy/10)
#    print ("F1 score ", F1/10)
    return accuracy/10


data_folder = str('DataFolder')
alpha = float(0.001)
n_epoch = int(50)
size = int(10000)
feature_type = str('type1')
reg = 1.0
lam = 1

#F1_tr_rate = list()
#F1_te_rate = list()
#
#rate = np.array([0.0001,0.01,0.1])

#data_folder = str(sys.argv[3])
#feature_type = str(sys.argv[2])
## transfer reg to one
#if arg == "true" or "True" or "TRUE": reg = 1
#else: reg = 0

#Import traning dataset
filename = 'train-images-idx3-ubyte'
tr_data = get_file(data_folder,filename)
tr_data = tr_data[0:size]
tr_X = get_feature1(tr_data)

#Import training label
filename = 'train-labels-idx1-ubyte'
tr_label = get_file(data_folder,filename)
tr_y = np.reshape(tr_label[0:size],(size,1))

# Forming training dataset
train = np.hstack((tr_X,tr_y))

#Import test dataset
filename = 't10k-images-idx3-ubyte'
te_data = get_file(data_folder,filename)
te_X = get_feature1(te_data)

#Import test label
filename = 't10k-labels-idx1-ubyte'
te_label = get_file(data_folder,filename)
te_y = np.reshape(te_label,(len(te_label),1))

# Forming test dataset
test = np.hstack([te_X,te_y])

# Calculate weights

predictions_tr = perceptron(train, train, l_rate, n_epoch)
predictions_te = perceptron(train, test, l_rate, n_epoch)

F1_tr,acc_tr = evaluate(predictions_tr,tr_y)
F1_te,acc_te = evaluate(predictions_te,te_y)


#
#    F1_tr_rate.append(F1_tr)
#    F1_te_rate.append(F1_te)

print ('Epoch %d: Training loss: %s, Training Accurary: %s, Test Accuracy: %s' %n_epoch, %acc_tr, %acc_te)
