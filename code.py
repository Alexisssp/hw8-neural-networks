#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 21:05:28 2019

@author: alexissaint-pierre
"""



#######question 4-7##########
#from __future__ import print_function
#import torch
#import numpy as np
#
#import load_data
#from my_dataset import MyDataset
#from torch.utils.data import Dataset
#from torch import nn
#from torch.nn import MSELoss
#from torch.optim import SGD
#from torch.nn.modules.loss import CrossEntropyLoss
#import torch
#import numpy as np
#import matplotlib.pyplot as plt
#from torch.utils.data.dataset import Dataset
#import torch.nn as nn
#import torch.nn.functional as F
#import torch.optim as optim
#import time
#
#class Network(nn.Module):
#    def __init__(self):
#        super(Network, self).__init__()
#        self.input = nn.Linear(784, 128)
#        self.hidden = nn.Linear(128, 64)
#        self.output = nn.Linear(64, 10)
#
#    def forward(self, x):
#        x = F.relu(self.input(x))
#        x = F.relu(self.hidden(x))
#        x = F.softmax(self.output(x))
#        return x
#    

#test_data, test_labels = load_data._load_mnist(path = "mnist", dataset = "testing")
#test_data=test_data.reshape(10000,784)
#times=[]
#ACRS=[]
#setofdata=[50,100,150,200]
#for k in range(len(setofdata)) :    
#    train_features,test_features,train_targets, test_targets = load_data.load_mnist_data(10, examples_per_class=setofdata[k], mnist_folder= "mnist")
#    train_dataset=MyDataset(train_features, train_targets)
#    network=Network().double()
#    criterion=nn.CrossEntropyLoss()
#    optimizer=optim.SGD(network.parameters(), lr=0.01)
#    listofloss=[]
#    now = time.time()
#    for epoch in range(100):  
#        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle = True)
#        run_loss = 0.0
#        for setofdata[k], data in enumerate(trainloader):
#            inputs, labels = data
#            optimizer.zero_grad()
#            outputs = network(inputs)
#            loss = criterion(outputs, labels.to(dtype = torch.long))
#            loss.backward()
#            optimizer.step()           
#            run_loss += loss.item()
#        listofloss.append(run_loss)
#        run_loss = 0.0
#
#        images, labels = test_data, test_labels
#        images = torch.from_numpy(images)
#        labels = torch.from_numpy(labels)
#        outputs = network(images)
#        _, predicted = torch.max(outputs.data, 1)
#        total = labels.shape[0]
#       
#    later = time.time()
#    difference = later - now
#    times.append(difference)
#    
#fig = plt.figure()
#
#plt.plot(setofdata, times, color='r')
#plt.xlabel("Number of training examples")
#plt.ylabel("Amount of training time (s)")
#
#plt.savefig("4.png",dpi=400)
#plt.show()
#
#
#fig = plt.figure()
#
#plt.plot(setofdata, ACRS, color='r')
#plt.xlabel("Number of training examples")
#plt.ylabel("accuracy")
#plt.savefig("6.png",dpi=400)
#plt.show()


#######question 8-12##########

#import numpy as np
#import pandas as pd
#from matplotlib.pyplot import imread
#import os
#import load_data
#from my_dataset import MyDataset
#from dogs import DogsDataset
#
#import torch
#from torch.utils.data import Dataset
#from torch import nn
#from torch.nn import MSELoss
#from torch.optim import SGD
#from torch.nn.modules.loss import CrossEntropyLoss
#import torch
#import matplotlib.pyplot as plt
#from torch.utils.data.dataset import Dataset
#import torch.nn.functional as F
#
#new_data = DogsDataset("Dogset")
#train_features, train_labels = new_data.get_train_examples()
#test_features, test_labels = new_data.get_test_examples()
#val_features, val_labels = new_data.get_validation_examples()

#class Network(nn.Module):
#    def __init__(self):
#        super(Network, self).__init__()
#        self.input = nn.Linear(12288, 128)
#        self.hidden = nn.Linear(128, 64)
#        self.output = nn.Linear(64, 10)
#
#    def forward(self, x):
#        x=F.relu(self.input(x))
#        x=F.relu(self.hidden(x))
#        x=F.softmax(self.output(x))
#        return x
#
#times=[]
#ACRS=[]
#
#
#train_dataset=MyDataset(train_features.reshape(7665,12288), train_labels)
#
#
#network = Network().double()
#criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(network.parameters(), lr=0.001)
#
#
#
#
#now = time.time()
#    
#accuracy_old = 0 
#
#listofloss_training = []
#ACRS_validation = []
#listofloss_validation = []
#epochh=[]
#
#for epoch in range(100):
#    print("epoch: " + str(epoch))
#    epochh.append(epoch)
#    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=32)
#    run_loss = 0.0
#    for i, data in enumerate(trainloader):
#        inputs, labels = data
#        optimizer.zero_grad()
#        outputs = network(inputs.to(dtype = torch.double))
#        loss = criterion(outputs, labels.to(dtype = torch.long))
#        loss.backward()
#        optimizer.step()
#
#        # print statistics
#        run_loss += loss.item()
#        
#    listofloss_training.append(run_loss)
#    with torch.no_grad():
#
#        images, labels = val_features.reshape(2000,12288), val_labels
#        images = torch.from_numpy(images)
#        labels = torch.from_numpy(labels)
#        outputs = network(images.to(dtype = torch.double))
#        _, predicted = torch.max(outputs.data, 1)
#            
#    if epoch%3==0:
#        if accuracy-accuracy_old < 1e-4:
#            break      
#        accuracy_old = accuracy
#
#with torch.no_grad():
#    
#    images, labels = test_features.reshape(555,12288), test_labels
#    images = torch.from_numpy(images)
#    labels = torch.from_numpy(labels)
#    outputs = network(images.to(dtype = torch.double))
#    _, predicted = torch.max(outputs.data, 1)
#later = time.time()
#difference = later - now
#times.append(difference)

#fig, ax1 = plt.subplots()
#ax1.plot(epochh, listofloss_training, 'b-')
#ax1.set_xlabel('Epoch')
#ax1.set_ylabel('Training loss', color='b')
#ax1.tick_params('y', colors='b')
#
#ax2 = ax1.twinx()
#ax2.plot(epochh, listofloss_validation, 'r')
#ax2.set_ylabel('Validation loss', color='r')
#ax2.tick_params('y', colors='r')
#
#fig.tight_layout()
#plt.savefig('11.png',dpi=400)
#plt.show()
#
#plt.plot(epochh,ACRS_validation, color='r')
#plt.xlabel("Epoch")
#plt.ylabel("Validation Accuracy")
#plt.savefig('112.png', dpi=400)


######question 14-15######



import numpy as np
import pandas as pd
from matplotlib.pyplot import imread
import os
import load_data
from my_dataset import MyDataset
from dogs import DogsDataset

import torch
from torch.utils.data import Dataset
from torch import nn
from torch.nn import MSELoss
from torch.optim import SGD
from torch.nn.modules.loss import CrossEntropyLoss
import torch
import matplotlib.pyplot as plt
from torch.utils.data.dataset import Dataset
import torch.nn.functional as F

new_data = DogsDataset("Dogset")

train_features, train_labels = new_data.get_train_examples()
test_features, test_labels = new_data.get_test_examples()
val_features, val_labels = new_data.get_validation_examples()

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        #Convolution 1
        self.cnn1 = nn.Conv2d(in_channels=12288, out_channels=32, kernel_size=3, stride=8, padding=4)
        self.relu1 = nn.ReLU()
        
        #Convolution 2
        self.cnn2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=8, padding=4)
        self.relu2 = nn.ReLU()
        
        
        #Dropout for regularization
        self.dropout = nn.Dropout(p=0.5)
        
        #Fully Connected 1
        self.fc1 = nn.Linear(64, 10)
        

    def forward(self, x):
        
        out = self.cnn1(x)
        out = self.relu1(out)

        
        #Convolution 2
        out = self.cnn2(out)
        out = self.relu2(out)
        
        #Resize
        out = out.view(out.size(0), -1)
        
        #Dropout
        out = self.dropout(out)
        
        #Fully connected 1
        out = self.fc1(out)
        return out

import torch.optim as optim
import time

#Loading the data    

times = []
ACRS = []


train_dataset = MyDataset(train_features.reshape(7665,12288), train_labels)


network = Network().double()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(network.parameters(), lr=0.001)




now = time.time()
    
accuracy_old = 0 

listofloss_training = []
ACRS_validation = []
listofloss_validation = []
epochh=[]

for epoch in range(100):
    print("epoch: " + str(epoch))
    epochh.append(epoch)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=32)
    run_loss = 0.0
    for i, data in enumerate(trainloader):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = network(inputs.to(dtype = torch.double))
        loss = criterion(outputs, labels.to(dtype = torch.long))
        loss.backward()
        optimizer.step()
        run_loss += loss.item()
        
    listofloss_training.append(run_loss)



    with torch.no_grad():

        images, labels = val_features.reshape(2000,12288), val_labels
        images = torch.from_numpy(images)
        labels = torch.from_numpy(labels)
        outputs = network(images.to(dtype = torch.double))
        _, predicted = torch.max(outputs.data, 1)
        total = labels.shape[0]

    if epoch%3 == 0:
        if accuracy - accuracy_old < 1e-4:
            break      
        accuracy_old = accuracy

with torch.no_grad():
    
    images, labels = test_features.reshape(555,12288), test_labels
    images = torch.from_numpy(images)
    labels = torch.from_numpy(labels)
    outputs = network(images.to(dtype = torch.double))
    _, predicted = torch.max(outputs.data, 1)


later = time.time()
difference = later - now
times.append(difference)

fig, ax1 = plt.subplots()
ax1.plot(epochh, listofloss_training, 'b-')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Training loss', color='b')
ax1.tick_params('y', colors='b')

ax2 = ax1.twinx()
ax2.plot(epochh, listofloss_validation, 'r')
ax2.set_ylabel('Validation loss', color='r')
ax2.tick_params('y', colors='r')

fig.tight_layout()
plt.savefig('131.png',dpi=400)
plt.show()

plt.plot(epochh,ACRS_validation, color='r')
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.savefig('132.png', dpi=400)



