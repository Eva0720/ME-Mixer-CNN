import os
import random
import cv2
import pandas
import matplotlib
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from natsort import natsorted
from PIL import Image
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, Conv2D, MaxPool2D,LayerNormalization
from tensorflow.keras.layers import ZeroPadding2D, AveragePooling2D, GlobalAveragePooling2D,GlobalMaxPooling2D
from tensorflow.keras.layers import Dropout, Activation, BatchNormalization, LeakyReLU
from tensorflow.python.keras.layers.core import Flatten
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.python.keras.utils import np_utils
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from tensorflow.keras.models import load_model
import tkinter
from PIL import Image
#from keras.utils.generic_utils import get_custom_objects
from tensorflow.keras.utils import get_custom_objects
#from keras_layer_normalization import LayerNormalization
    

def custom_gelu(x):
    return 0.5*x*(1+tf.tanh(np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x, 3))))
get_custom_objects().update({'custom_gelu': Activation(custom_gelu)})

def mlp_block(x, mlp_dim):
    x = layers.Dense(mlp_dim, activation=custom_gelu)(x)
    return layers.Dense(int(x.shape[-1]))(x)


def mixer_block(x, tokens_mlp_dim, channels_mlp_dim):
    y = LayerNormalization()(x)
    y = layers.Permute((2, 1))(y)

    token_mixing = mlp_block(y, tokens_mlp_dim)
    token_mixing = layers.Permute((2, 1))(token_mixing)
    x = layers.Add()([x, token_mixing])
    y = LayerNormalization()(x)
    channel_mixing = mlp_block(y, channels_mlp_dim)
    output = layers.Add()([x, channel_mixing])
    return output

def mlp_mixer(x, num_blocks, patch_size, hidden_dim,
              tokens_mlp_dim, channels_mlp_dim,out_dim):
    x = layers.Conv2D(hidden_dim, kernel_size=patch_size,
                      strides=patch_size, padding="valid")(x)
    x = layers.Reshape((int(x.shape[1]) * int(x.shape[2]), int(x.shape[3])))(x)

    for _ in range(num_blocks):
        x = mixer_block(x, tokens_mlp_dim, channels_mlp_dim)

    x = LayerNormalization()(x)
    x = layers.Dropout(0.25)(x)
    x = layers.GlobalAveragePooling1D()(x)
  #  return layers.Dense(num_classes, activation="softmax", dtype="float32")(x)
    return layers.Dense(out_dim,activation="relu")(x)
    
def mlp_mixer_corr(x, num_blocks, patch_size,
              tokens_mlp_dim, channels_mlp_dim,
              output_dim):

    x = layers.Reshape((int(x.shape[1]) * int(x.shape[2]), int(x.shape[3])))(x)

    for _ in range(num_blocks):
        x = mixer_block(x, tokens_mlp_dim, channels_mlp_dim)

    x = LayerNormalization()(x)
    x = layers.Dropout(0.25)(x)
    x = layers.GlobalAveragePooling1D()(x)
    return layers.Dense(output_dim, activation="relu", dtype="float32")(x)  
      
    
def _mlp(x,channel):
  x=tf.reshape(x,shape=(-1,1,1,channel))
  x=Dense(channel//8,kernel_initializer='he_normal',activation = tf.nn.swish,use_bias=True, bias_initializer='zeros')(x)
  x=Dense(channel,kernel_initializer='he_normal',use_bias=True,bias_initializer='zeros')(x)
  return x
    