import pickle
import numpy as np
from sklearn.metrics import accuracy_score
import pandas as pd
import datetime
import time


def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter


def load_data(path='/work/s6kalra/'):
    """
    Load data
    """
    with open('{0}/train_data'.format(path), 'rb') as f:
        train_data = pickle.load(f)
        train_label = pickle.load(f)

    with open('{0}/test_data'.format(path), 'rb') as f:
        test_data = pickle.load(f)

    x_train = np.array([
        np.rot90(np.fliplr(img.reshape((3, 32, 32)).T)) for img in train_data
    ])
    x_train = x_train.astype('uint8')

    x_test = np.array(
        [np.rot90(np.fliplr(img.reshape((3, 32, 32)).T)) for img in test_data])
    x_test = x_test.astype('uint8')

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    # print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = np.eye(100)[train_label]

    return {
        'x_train': x_train,
        'y_train': y_train,
        'y_labels': train_label,
        'x_test': x_test
    }


def predict_score(predictions, path='/work/s6kalra', is_labels=False):
    with open('{0}/test'.format(path), 'rb') as f:
        test2_data = pickle.load(f, encoding='bytes')
    orlabels = test2_data[b'fine_labels']

    if not is_labels:
        predictions = np.argmax(predictions, axis=1)
    return accuracy_score(predictions, orlabels)


def get_true_labels(path='/work/s6kalra'):
    with open('{0}/test'.format(path), 'rb') as f:
        test2_data = pickle.load(f, encoding='bytes')
    return test2_data[b'fine_labels']


def write_results(predictions, path='.', is_labels=False):
    if not is_labels:
        predictions = np.argmax(predictions, axis=1)

    df = pd.DataFrame(predictions)
    df.index.name = 'ids'
    df.columns = ['labels']
    df.to_csv(
        '{0}/results_{1}.csv'.format(
            path,
            datetime.datetime.fromtimestamp(
                time.time()).strftime('%Y-%m-%d-%H:%M:%S')),
        index=True)
