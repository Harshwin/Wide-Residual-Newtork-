
# coding: utf-8

# In[4]:


import lib
import wide_residual_network as wrn


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

if lib.isnotebook():
    from keras_tqdm import TQDMNotebookCallback as KerasCallBack
else: 
    from keras_tqdm import TQDMCallback as KerasCallBack


# In[5]:


print('Building model...')
K.set_image_data_format('channels_last')

# wrn 28 10
model = wrn.create_wide_residual_network((32, 32, 3), nb_classes=100, N=5, k=10, dropout=0.3, verbose=1)


# In[6]:


print('Loading data...')
data = lib.load_data()
x_train = data['x_train']
y_train = data['y_train']


# In[16]:


x_train.shape, y_train.shape


# In[17]:


print('Fitting augementer...')
# Augement the images
datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=0,
    width_shift_range=0.0,
    height_shift_range=0.0,
    horizontal_flip=True,
    data_format=K.image_data_format())

# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
datagen.fit(x_train)


# In[18]:


print('Compiling model...')
# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# In[ ]:


print('Training model...')
# fits the model on batches with real-time data augmentation:
model.fit_generator(datagen.flow(x_train, y_train, batch_size=64),
                    steps_per_epoch=int(len(x_train) / 64), epochs=100, 
                    verbose=0, callbacks=[KerasCallBack()])

model.save('wrn28-10')

