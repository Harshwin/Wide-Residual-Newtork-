
# coding: utf-8

# In[1]:


import lib
import stack_image as si
import glob
import cv2
import os.path

# In[2]:


import wide_residual_network as wrn
import irmalib as irma


#import matplotlib as mpl
#import matplotlib.pyplot as plt

import tensorflow as tf

import numpy as np
import tensorflow as tf
from keras.datasets import cifar10
from keras.layers import Dense, Activation, Flatten, Lambda, Conv2D, AvgPool2D, BatchNormalization, Dropout, merge
from keras.engine import Input, Model
from keras.optimizers import SGD
from keras.callbacks import Callback, LearningRateScheduler, ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
import keras.backend as K
from keras.models import load_model
import json
import time
from keras.utils import multi_gpu_model
#if lib.isnotebook():
#    from keras_tqdm import TQDMNotebookCallback as KerasCallBack
#else: 
from keras_tqdm import TQDMCallback as KerasCallBack


# In[3]:


data = irma.load_data('./IRMA/')
paths = data['train_data']['image_path']

train_dataset = np.array([np.expand_dims(irma.imread_norm(path, (112, 112)), axis=2) for path in paths])

path1 = data['test_data']['image_path']
test_dataset = np.array([np.expand_dims(irma.imread_norm(path, (112, 112)), axis=2) for path in path1])

# In[4]:


train_labels = data['train_data']['irma_05_code']
test_labels = data['test_data']['irma_05_code']

# In[5]:


train_lbls = np.eye(max(train_labels)+1)[train_labels]
test_lbls = np.eye(max(test_labels)+1)[test_labels]

# In[6]:


train_lbls[0].shape


# In[7]:


np.expand_dims(irma.imread_norm(paths[0]), axis=2).shape
np.expand_dims(irma.imread_norm(path1[0]), axis=2).shape

print('Building model...')
K.set_image_data_format('channels_last')

# wrn 28 10
model = wrn.create_wide_residual_network((112, 112, 1), nb_classes=57, N=5, k=10, dropout=0.3, verbose=1)

import multi_gpu as mg
model = multi_gpu_model(model, gpus=8)


print('Compiling model...')
sgd = SGD(lr=0.1, decay=0.0005, momentum=0.9)

class LrReducer(Callback):
    def __init__(self, epoch_val=0, reduce_rate=0.7):
        super(Callback, self).__init__()
        self.epoch_val = epoch_val
        self.reduce_rate = reduce_rate

        if os.path.isfile('epoch_values'):
            self.values = np.load('epoch_values')
        else:
            self.values = []

    def on_epoch_end(self, epoch, logs={}):
        self.epoch_val += 1
        current_score = logs.get('val_acc')
        train_score = logs.get('acc')

        #self.values.append([train_score, current_score])
        #np.save('epochs_values', self.values)
        if self.epoch_val % 60 == 0:
            print("Doing lr change")
            lr = K.get_value(self.model.optimizer.lr)
            K.set_value(self.model.optimizer.lr, lr * self.reduce_rate)


# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])


# checkpoint
filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max', save_weights_only=True)

kcall = KerasCallBack()

def run_from(n_epoch=0):
    lr_reducer = LrReducer(epoch_val=n_epoch)

    print('Training model...')
    model.fit(train_dataset, train_lbls, batch_size=8*10, 
              validation_data=(test_dataset, test_lbls), epochs=170, 
              callbacks=[kcall, checkpoint, lr_reducer])

run_from(0)

