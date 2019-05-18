
# coding: utf-8

# In[1]:


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

import imgaug as ia
from imgaug import augmenters as iaa

if lib.isnotebook():
    from keras_tqdm import TQDMNotebookCallback as KerasCallBack
else: 
    from keras_tqdm import TQDMCallback as KerasCallBack


# In[16]:


data = lib.load_data()


# In[155]:


seq = iaa.Sequential(
    [
        iaa.Fliplr(0.5),
        iaa.Sometimes(0.5,
            [iaa.Crop(px=(0, 12)),
            iaa.Pad(px=4, pad_mode=ia.ALL, pad_cval=(0, 255))]
        ),
        iaa.Sometimes(0.15,
            iaa.Dropout(p=0.05)
        )
    ],
    random_order=True
)


# In[156]:


def preprocess(images):
    return images.astype('float32')/255
    # return 2. * ((images.astype('float32') / 255) - .5)
    
def augmented_gen(batch_size=128):
    while True:
        for i in range(0, len(data['x_train']), batch_size):
            begin = i
            if i + batch_size > len(data['x_train']):
                begin = len(data['x_train']) - batch_size
                
            yield (preprocess(seq.augment_images(data['x_train'][i:i+batch_size])), 
            data['y_train'][i:i+batch_size])


# In[197]:


print('Building model...')
K.set_image_data_format('channels_last')

# wrn 28 10
model = wrn.create_wide_residual_network((32, 32, 3), nb_classes=100, N=5, k=10, dropout=0.3, verbose=1)


# In[198]:


x_test = preprocess(data['x_test'])
y_test = np.eye(100)[lib.get_true_labels()]


# In[200]:


sgd = SGD(lr=0.1, decay=0.0005, momentum=0.9)

class LrReducer(Callback):
    def __init__(self, patience=5, reduce_rate=0.8):
        super(Callback, self).__init__()
        self.patience = patience
        self.wait = 0
        self.best_score = -1.
        self.epoch_val = 0
        self.reduce_rate = reduce_rate
        self.history = []

    def on_epoch_end(self, epoch, logs={}):
        self.epoch_val += 1
        self.history.append(logs)
        
        current_score = logs.get('val_acc')
        if current_score > self.best_score:
            self.best_score = current_score
            self.wait = 0
        else:
            self.model.save('wrn28-10.aug.{0}.{1:.2f}'.format(self.epoch_val, current_score), overwrite=True)
            if self.wait >= self.patience:
                lr = self.model.optimizer.lr.get_value()
                self.model.optimizer.lr.set_value(lr*self.reduce_rate)
            self.wait += 1
            
print('Compiling model...')
# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])


# In[201]:


model.fit_generator(augmented_gen(),
                    steps_per_epoch=int(len(data['x_train']) / 128), 
                    epochs=300, 
                    validation_data=(x_test, y_test),
                    callbacks=[LrReducer()])

model.save('wrn28-10.aug.last')

