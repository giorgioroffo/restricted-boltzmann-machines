# Boltzmann Machines

# Importing the libraries
import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
from utils import convert, torch_like_notLike_encoding
from architectures import RBM

print('-----------------------------------------------------')
print('RBM-based Recommender System on MovieLens 1M Dataset ')
print('      https://grouplens.org/datasets/movielens/  ')
print('-----------------------------------------------------')

print('<Info> Importing the dataset')
# Importing the dataset
movies = pd.read_csv('ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
users = pd.read_csv('ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

print('<Info> Preparing the training set and the test set')
# Preparing the training set and the test set
training_set = pd.read_csv('ml-100k/u1.base', delimiter = '\t')
training_set = np.array(training_set, dtype = 'int')
test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t')
test_set = np.array(test_set, dtype = 'int')

print('<Info> Getting the number of users and movies')
# Getting the number of users and movies
nb_users = int(max(max(training_set[:,0]), max(test_set[:,0])))
nb_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))

print('<Info> Converting the data into an array with users in lines and movies in columns')
# Converting the data into an array with users in lines and movies in columns
training_set = convert(training_set, nb_users, nb_movies)
test_set = convert(test_set, nb_users, nb_movies)

print('<Info> Converting the ratings into binary ratings 1 (Liked) or 0 (Not Liked)')
# Converting the ratings into binary ratings 1 (Liked) or 0 (Not Liked)
training_set, test_set = torch_like_notLike_encoding(training_set, test_set)

print('<Info> Creating the architecture of the Neural Network')
# Creating the architecture of the Neural Network
nv = len(training_set[0])
nh = 100
batch_size = 100
rbm = RBM(nv, nh)

print('<Info> Training the RBM')
# Training the RBM
nb_epoch = 10
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0.
    for id_user in range(0, nb_users - batch_size, batch_size):
        # see: Algorithm 1. k-step contrastive divergence 
        # Fischer, A. and Igel, C., 2012, September. 
        # An introduction to restricted Boltzmann machines. 
        # In Iberoamerican congress on pattern recognition (pp. 14-36). 
        # Springer, Berlin, Heidelberg.
        vk = training_set[id_user:id_user+batch_size]
        v0 = training_set[id_user:id_user+batch_size]
        ph0,_ = rbm.sample_h(v0)
        for k in range(10):
            _,hk = rbm.sample_h(vk)
            _,vk = rbm.sample_v(hk)
            # do not include or update nodes with no rating == -1
            vk[v0<0] = v0[v0<0] 
        phk,_ = rbm.sample_h(vk)
        rbm.train(v0, vk, ph0, phk)
        # compute the loss on only RATED elements
        train_loss += torch.mean(torch.abs(v0[v0>=0] - vk[v0>=0]))
        s += 1.
    print('<Info> epoch: '+str(epoch)+' loss: '+str(train_loss/s))

print('<Info> Testing the RBM')
# Testing the RBM
test_loss = 0
s = 0.
for id_user in range(nb_users):

    # Use the training set to activate neurons  
    v = training_set[id_user:id_user+1]

    vt = test_set[id_user:id_user+1]

    # Make only 1 step in a blind walk on the Markov Chain
    # We only want to predict the RATED items >0
    if len(vt[vt>=0]) > 0:
        _,h = rbm.sample_h(v)
        _,v = rbm.sample_v(h)
        test_loss += torch.mean(torch.abs(vt[vt>=0] - v[vt>=0]))
        s += 1.
        
print('<Info> Test loss: '+str(test_loss/s))
print('<Info> Completed.')
