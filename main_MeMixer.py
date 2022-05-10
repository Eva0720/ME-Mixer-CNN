
import random
import cv2
import time
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
from tensorflow.keras.layers import Dense, Input, Conv2D, MaxPool2D
from tensorflow.keras.layers import ZeroPadding2D, AveragePooling2D, GlobalAveragePooling2D
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
from collections import Counter
from tensorflow.keras.applications import ResNet50,DenseNet121, MobileNetV2
from tensorflow.keras.applications.vgg16 import VGG16
from utils import *
import sklearn.metrics as metrics
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.metrics import categorical_accuracy
from focal_loss import *
from sklearn.utils import class_weight
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
matplotlib.use('AGG')
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

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def fbeta_score(y_true, y_pred, beta=1):
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')


    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score

def fmeasure(y_true, y_pred):
    return fbeta_score(y_true, y_pred, beta=1)

# load label
X_all=np.load('.../X.npy',allow_pickle=True)
X_all = np.asarray(X_all).astype(np.float32)
y_all=np.load('.../y.npy',allow_pickle=True)
y_all = np.asarray(y_all).astype(np.float32)

X_train, X_test1,y_train, y_test1 = train_test_split(X_all, y_all, test_size=0.2, random_state=0)
X_val, X_test,y_val, y_test = train_test_split(X_test1, X_test1, test_size=0.5, random_state=0)

X_train = X_train/255.0
X_val = X_val/255.0
X_test = X_test/255.0

batch_size = 32

X_train=X_train[0:(len(X_train)//batch_size)*batch_size]
X_val=X_val[0:(len(X_val)//batch_size)*batch_size]
X_test=X_test[0:(len(X_test)//batch_size)*batch_size]
y_train=y_train[0:(len(X_train)//batch_size)*batch_size]
y_val=y_val[0:(len(y_val)//batch_size)*batch_size]
y_test=y_test[0:(len(X_test)//batch_size)*batch_size]


weight_decay=0.0001
patch_size_mixer = 32
mixer_layer_number = 16
hidden_size = 128
seq_dim = 256
channel_dim = 128

# Model configuration
utsne_model=load_model('.../model_MEN.h5', custom_objects={'KLdivergence':KLdivergence})
low_para_model = Model(inputs=utsne_model.input,outputs=utsne_model.get_layer('encoded').output)
low_para_model.trainable = False
#loss_function = sparse_categorical_crossentropy
nb_epoch = 300
validation_split = 0.2
verbosity = 1
image_size =256
#establish model
shape1 = (256,256,3)
n_channels1 = 64
n_channels2 = 1
n_output = 2
input1 = Input(shape=shape1)
input2=Input(shape=(64,64,32))

base_model=ResNet50(include_top=False,weights='imagenet',input_shape=(image_size,image_size,3))
base_in=base_model.input
base_out=base_model.output
x=GlobalAveragePooling2D()(base_out)
output1 = Dense(1024,activation="relu")(x)


block5_conv3 = base_model.get_layer("conv5_block3_2_conv").output
tokens_inter_dim1=64
channel_inter1=512
inter1 = mlp_mixer_corr(block5_conv3, mixer_layer_number,patch_size_mixer,tokens_inter_dim1, channel_inter1,output_dim=512)

tokens_inter_dim2=256
channel_inter2=512
utsne_in=low_para_model.input
utsne_out=low_para_model.output
utsne = layers.Conv2D(512, kernel_size=4,strides=4, padding="valid", name='utsne_conv2d')(utsne_out)
utsne = mlp_mixer_corr(utsne, mixer_layer_number,patch_size_mixer,tokens_inter_dim2, channel_inter2,output_dim=256)

combine1 = Concatenate()([inter1,utsne])

combine1 = Dense(1024, kernel_initializer="he_normal",activation = 'relu', kernel_regularizer=l2(weight_decay))(combine1)
output = Concatenate()([output1, combine1])
output = Dense(1024, kernel_initializer="he_normal",activation = 'relu', kernel_regularizer=l2(weight_decay))(output)
output = Dense(512, kernel_initializer="he_normal",activation = 'relu', kernel_regularizer=l2(weight_decay))(output)
output = Dense(256, kernel_initializer="he_normal",activation = 'relu', kernel_regularizer=l2(weight_decay))(output)
out =Dense(8, activation='softmax')(output)
model = Model([base_model.input,low_para_model.input], out)

  
best_model_path='.../results/model_memixer.h5'
learning_rate=0.00005
model.compile(optimizer=Adam(learning_rate,decay=0.00001),loss='categorical_crossentropy', metrics=['acc',tf.keras.metrics.Precision(name='precision'),tf.keras.metrics.Recall(name='recall'),fmeasure,tf.keras.metrics.AUC(name='auc')], run_eagerly=True)
es=EarlyStopping(monitor='val_acc',
                        patience=30,verbose=1, mode='max')
# Reduce=ReduceLROnPlateau(monitor='val_acc',
#                          factor=0.8,
#                          patience=10,
#                          verbose=1,
#                          mode='auto',
#                          epsilon=0.00001,
#                          cooldown=0,
#                          min_lr=0)
mc = ModelCheckpoint(best_model_path, monitor='val_acc', save_best_only=True, mode='max')


def generator_train(X1,y, batch_size):
    gen = ImageDataGenerator(      rotation_range=90,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.4,
                                   zoom_range=0.4,
                                   horizontal_flip=True,
                                   fill_mode='nearest')
    genX1 = gen.flow(X1, y,  batch_size=batch_size, seed=1)
  #  genX2 = gen.flow(X1, X2, batch_size=batch_size, seed=1)
    while True:
        X1i = genX1.next()
     #   X2i = genX2.next()
        yield [X1i[0], X1i[0]], X1i[1]
        
def generator_test(X1, y, batch_size):
    gen = ImageDataGenerator()
    genX1 = gen.flow(X1, y,  batch_size=batch_size, seed=0)
   # genX2 = gen.flow(X1, X2, batch_size=batch_size, seed=1)
    while True:
        X1i = genX1.next()
      #  X2i = genX2.next()
        yield [X1i[0], X1i[0]], X1i[1]         
         
# Fit data to model
history=model.fit(generator_train(X_train,y_train, batch_size),
                    steps_per_epoch=len(X_train) // batch_size, epochs=400,
                    validation_data = generator_test(X_val,y_val,batch_size),
                    validation_steps = len(X_val)//batch_size,
                    callbacks=[es,mc])


# Evaluate
model=load_model(best_model_path, custom_objects={'fmeasure':fmeasure})
score = model.evaluate([X_test,X_test], verbose=0)
print(f'1branch_Test loss: {score[0]} / Test Accuracy: {score[1]}  / Test Precision: {score[2]}  / Test Recall: {score[3]}  / Test F_score: {score[4]} / Test auc: {score[5]} ')

pred_all=model.predict([X_test,X_test])
y_pred = np.argmax(pred_all,axis=1)
y_test = np.argmax(y_test,axis=1)
print(classification_report(y_test[0:batch_size*(len(X_test)//batch_size)], y_pred[0:batch_size*(len(X_test)//batch_size)],digits=4))


