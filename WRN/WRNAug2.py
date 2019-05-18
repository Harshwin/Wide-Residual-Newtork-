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
        iaa.Sometimes(
            0.5, [
                iaa.Crop(px=(0, 12)),
                iaa.Pad(px=4, pad_mode=ia.ALL, pad_cval=(0, 255))
            ]),
        iaa.Sometimes(0.15, iaa.Dropout(p=0.05))
    ],
    random_order=True)

# In[156]:


def preprocess(images):
    return images.astype('float32') / 255
    # return 2. * ((images.astype('float32') / 255) - .5)


# In[197]:

print('Building model...')
K.set_image_data_format('channels_last')

# wrn 28 10

# model = wrn.create_wide_residual_network(
#     (32, 32, 3), nb_classes=100, N=5, k=10, dropout=0.3, verbose=1)
model = load_model('wrn28-10.aug.123.0.81')

# In[198]:

x_test = preprocess(data['x_test'])
y_test = np.eye(100)[lib.get_true_labels()]


def augmented_gen(batch_size=128):
    while True:
        shuf_idx = list(range(0, len(data['x_train'])))
        np.random.shuffle(shuf_idx)
        shuf_x = data['x_train'][shuf_idx]
        shuf_y = data['y_train'][shuf_idx]
        for i in range(0, len(shuf_x), batch_size):
            begin = i
            if i + batch_size > len(shuf_x):
                begin = len(shuf_x) - batch_size

            yield (preprocess(
                seq.augment_images(shuf_x[begin:begin + batch_size])),
                   shuf_y[begin:begin + batch_size])


# In[200]:
lr = 0.1 * 0.1
sgd = SGD(lr=lr, decay=0.0005, momentum=0.9)


class LrReducer(Callback):
    def __init__(self, epoch_val=0, reduce_rate=0.7):
        super(Callback, self).__init__()
        self.epoch_val = epoch_val
        self.reduce_rate = reduce_rate
        self.history = []
        self.best_score = -1
        self.values = list(np.load('epochs_values.npy'))

    def on_epoch_end(self, epoch, logs={}):
        self.epoch_val += 1
        self.history.append(logs)

        current_score = logs.get('val_acc')
        train_score = logs.get('acc')

        self.model.save(
            'wrn28-10.aug.{0}.{1:.2f}'.format(self.epoch_val, current_score),
            overwrite=True)

        self.values.append([train_score, current_score])
        np.save('epochs_values', self.values)
        if current_score > self.best_score:
            self.best_score = current_score

        if self.epoch_val % 60 == 0:
            print("Doing lr change")
            lr = K.get_value(self.model.optimizer.lr)
            K.set_value(self.model.optimizer.lr, lr * self.reduce_rate)


print('Compiling model...')
# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
model.compile(
    loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# In[201]:

reducer = LrReducer(epoch_val=123)
model.fit_generator(
    augmented_gen(),
    steps_per_epoch=int(len(data['x_train']) / 128),
    epochs=300,
    validation_data=(x_test, y_test),
    callbacks=[reducer])

model.save('wrn28-10.aug.last')
