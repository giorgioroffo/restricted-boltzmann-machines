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

print('Importing the dataset')
# Importing the dataset
movies = pd.read_csv('ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
users = pd.read_csv('ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

print('Preparing the training set and the test set')
# Preparing the training set and the test set
training_set = pd.read_csv('ml-100k/u1.base', delimiter = '\t')
training_set = np.array(training_set, dtype = 'int')
test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t')
test_set = np.array(test_set, dtype = 'int')

print('Getting the number of users and movies')
# Getting the number of users and movies
nb_users = int(max(max(training_set[:,0]), max(test_set[:,0])))
nb_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))

print('Converting the data into an array with users in lines and movies in columns')
# Converting the data into an array with users in lines and movies in columns
training_set = convert(training_set, nb_users, nb_movies)
test_set = convert(test_set, nb_users, nb_movies)

print('Converting the ratings into binary ratings 1 (Liked) or 0 (Not Liked)')
# Converting the ratings into binary ratings 1 (Liked) or 0 (Not Liked)
training_set, test_set = torch_like_notLike_encoding(training_set, test_set)

print('Creating the architecture of the Neural Network')
# Creating the architecture of the Neural Network
nv = len(training_set[0])
nh = 100
batch_size = 100
rbm = RBM(nv, nh)

print('Training the RBM')
# Training the RBM
nb_epoch = 10
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0.
    for id_user in range(0, nb_users - batch_size, batch_size):
        vk = training_set[id_user:id_user+batch_size]
        v0 = training_set[id_user:id_user+batch_size]
        ph0,_ = rbm.sample_h(v0)
        for k in range(10):
            _,hk = rbm.sample_h(vk)
            _,vk = rbm.sample_v(hk)
            vk[v0<0] = v0[v0<0]
        phk,_ = rbm.sample_h(vk)
        rbm.train(v0, vk, ph0, phk)
        train_loss += torch.mean(torch.abs(v0[v0>=0] - vk[v0>=0]))
        s += 1.
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))

print('Testing the RBM')
# Testing the RBM
test_loss = 0
s = 0.
for id_user in range(nb_users):
    v = training_set[id_user:id_user+1]
    vt = test_set[id_user:id_user+1]
    if len(vt[vt>=0]) > 0:
        _,h = rbm.sample_h(v)
        _,v = rbm.sample_v(h)
        test_loss += torch.mean(torch.abs(vt[vt>=0] - v[vt>=0]))
        s += 1.
        
print('> Test loss: '+str(test_loss/s))
print('Completed.')
