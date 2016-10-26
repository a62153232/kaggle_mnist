# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 15:42:02 2016

@author: 114
"""

import pandas as pd
import numpy as np  
import lasagne  
from lasagne import layers  
from lasagne.updates import nesterov_momentum  
from nolearn.lasagne import NeuralNet

def load_mnist():
    inpath = "E:\\Anaconda_working\\kaggle_mnist\\train.csv"
    with open(inpath,'rb') as file:
        train=pd.read_csv(file)
        
    inpath = "E:\\Anaconda_working\\kaggle_mnist\\test.csv"
    with open(inpath,'rb') as file:
        test=pd.read_csv(file)
    
    X_train = train.iloc[:,1:].values 
    y_train = train.iloc[:,0].values
    X_test = test.values
    
    X_train = (1.00/255.00) * X_train.reshape((-1, 1, 28, 28)).astype(np.float32)  # 归一化
    X_test = (1.00/255.00) * X_test.reshape((-1, 1, 28, 28)).astype(np.float32)    
    y_train = y_train.astype(np.uint8) 
    
    return X_train, y_train, X_test

X_train, y_train, X_test = load_mnist()

net1 = NeuralNet(  
    layers=[('input', layers.InputLayer),  
            ('conv2d1', layers.Conv2DLayer),  
            ('maxpool1', layers.MaxPool2DLayer),  
            ('conv2d2', layers.Conv2DLayer),  
            ('maxpool2', layers.MaxPool2DLayer),  
            ('dropout1', layers.DropoutLayer),  
            ('dense', layers.DenseLayer),  
            ('dropout2', layers.DropoutLayer),  
            ('output', layers.DenseLayer),  
            ],  
    # input layer  
    input_shape=(None, 1, 28, 28),  
    # layer conv2d1  
    conv2d1_num_filters=32,  
    conv2d1_filter_size=(5, 5),  
    conv2d1_nonlinearity=lasagne.nonlinearities.rectify,  
    conv2d1_W=lasagne.init.GlorotUniform(),    
    # layer maxpool1  
    maxpool1_pool_size=(2, 2),      
    # layer conv2d2  
    conv2d2_num_filters=32,  
    conv2d2_filter_size=(5, 5),  
    conv2d2_nonlinearity=lasagne.nonlinearities.rectify,  
    # layer maxpool2  
    maxpool2_pool_size=(2, 2),  
    # dropout1  
    dropout1_p=0.5,      
    # dense  
    dense_num_units=256,  
    dense_nonlinearity=lasagne.nonlinearities.rectify,      
    # dropout2  
    dropout2_p=0.5,      
    # output  
    output_nonlinearity=lasagne.nonlinearities.softmax,  
    output_num_units=10,  
    # optimization method params  
    update=nesterov_momentum,  
    update_learning_rate=0.01,  
    update_momentum=0.9,
    max_epochs=100,  
    verbose=1,  
    )  
# Train the network  
nn = net1.fit(X_train, y_train)

pred = net1.predict(X_test)

#pred = pred.reshape((-1, 1))
##arr = np.arange(1,28001,dtype=np.int64).reshape((-1,1))
##csv = np.hstack((arr,pred))           
##csv = pd.DataFrame(csv)
##csv.columns = ['XXX','XXX']         #修改列名
##csv.to_csv("XXX.csv",index=False)   #不显示index列