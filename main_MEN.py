import tensorflow as tf
import math
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split

import numpy as np
np.random.seed(71)
import os
import random
import matplotlib
matplotlib.use('Agg')
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation

import multiprocessing as mp

from tensorflow.keras.models import Model

from tensorflow.keras.layers import (Conv2D, Activation, Dense, Lambda, Input,

    MaxPooling2D, Dropout, Flatten, Reshape, UpSampling2D, Concatenate)

from tensorflow.keras.losses import mse

from tensorflow.keras.utils import plot_model

from tensorflow.keras import backend as K
from tsne_utils import x2p
batch_size = 5000
low_dim = 2
nb_epoch = 100
shuffle_interval = nb_epoch + 1
n_jobs = 4
perplexity = 30.0

def calculate_P(X):
    n = X.shape[0]
    P = np.zeros([n, batch_size])
    for i in range(0, n, batch_size):
        P_batch = x2p(X[i:i + batch_size],perplexity)
        P_batch[np.isnan(P_batch)] = 0
        P_batch = P_batch + P_batch.T
        
                #exaggerate
        P_batch = P_batch*2
                
        P_batch = P_batch / P_batch.sum()
        P_batch = np.maximum(P_batch, 1e-12) 
        P[i:i + batch_size] = P_batch
    return P

def KLdivergence(P, Y):
    alpha = low_dim - 1.
    sum_Y = K.sum(K.square(Y), axis=1)
    eps = K.variable(10e-15)
    D = sum_Y + K.reshape(sum_Y, [-1, 1]) - 2 * K.dot(Y, K.transpose(Y))
    Q = K.pow(1 + D / alpha, -(alpha + 1) / 2)
    Q *= K.variable(1 - np.eye(batch_size))
    Q /= K.sum(Q)
    Q = K.maximum(Q, eps)
    C = K.log((P + eps) / (Q + eps))
    C = K.sum(P * C)
    return C

#GPU selection
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ[ 'CUDA_VISIBLE_DEVICES' ] = ''
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
my_seed = 0
os.environ['PYTHONHASHSEED'] = str(0)
random.seed(my_seed)
tf.random.set_seed(my_seed)
np.random.seed(7)
    
# load label
X_all=np.load('.../X.npy',allow_pickle=True)
X_all = np.asarray(X_all).astype(np.float32)
X_all=X_all/255.0


batch_size=512
X_train=X_all[0:(len(X_all)//batch_size)*batch_size]


n,image_size,image_size,channel= X_train.shape

batch_num = int(n // batch_size)
m = batch_num * batch_size

input_img = Input(shape=(image_size, image_size, channel))
x = Conv2D(8, (3, 3), activation='relu', padding='same')(input_img)  
x = MaxPooling2D((2, 2), padding='same',name='pooling1')(x)  
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)  
x = MaxPooling2D((2, 2), padding='same',name='pooling2')(x)  
encoded = Conv2D(32, (3, 3), activation='relu', padding='same',name='encoded')(x)   

low_dim_f=Flatten()(encoded)
low_dim1=Dense((500),name='low_inter')(low_dim_f)
low_dim2=Dense((100))(low_dim1)
low_dim3=Dense(2)(low_dim2)

x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)  
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)  
x = UpSampling2D((2, 2))(x)  
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)  
x = UpSampling2D((2, 2))(x)  
decoded = Conv2D(3, (3, 3), padding='same')(x)    

model=Model(inputs = [input_img], outputs = [low_dim3, decoded])
#
model.compile(
              optimizer='adam',
              loss=[KLdivergence,'mean_squared_error']
              )

model=load_model('.../model_MEN.h5', custom_objects={'KLdivergence':KLdivergence})
