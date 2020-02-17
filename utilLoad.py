# -*- coding: utf-8 -*-
from __future__ import print_function
import cv2
import numpy as np
from sklearn.utils import shuffle
import util
import utilLoadMNIST
import utilLoadSigns


# ----------------------------------------------------------------------------
def __run_validations(datasets):
    input_shape = None
    nb_classes = None
    assert datasets is not None
    assert len(datasets) > 0

    for i in range(len(datasets)):
        assert 'name' in datasets[i] and datasets[i]['name'] is not None
        assert 'x_train' in datasets[i] and datasets[i]['x_train'] is not None
        assert 'y_train' in datasets[i] and datasets[i]['y_train'] is not None
        assert 'x_test' in datasets[i] and datasets[i]['x_test'] is not None
        assert 'y_test' in datasets[i] and datasets[i]['y_test'] is not None

        if input_shape is None:
            input_shape = datasets[i]['x_train'].shape[1:]
        assert input_shape == datasets[i]['x_train'].shape[1:]
        assert input_shape == datasets[i]['x_test'].shape[1:]

        if nb_classes is None:
            nb_classes = len(datasets[i]['y_train'][1])
        assert nb_classes == len(datasets[i]['y_train'][1])
        assert nb_classes == len(datasets[i]['y_test'][1])

    return input_shape, nb_classes


# ----------------------------------------------------------------------------
def __smooth_labels(datasets):
    print('Smoothing labels...')
    for i in range(len(datasets)):
        datasets[i]['y_train'] = util.smooth_labels(datasets[i]['y_train'].astype('float32'))


# ----------------------------------------------------------------------------
def __normalize(datasets, norm_type):
    for i in range(len(datasets)):
        x_train = datasets[i]['x_train']
        x_test = datasets[i]['x_test']
        x_train = np.asarray(x_train).astype('float32')
        x_test = np.asarray(x_test).astype('float32')

        print('Normalize dataset', datasets[i]['name'])
        print(' - Min / max / avg train:', np.min(x_train), ' / ', np.max(x_train), ' / ', np.mean(x_train))
        print(' - Min / max / avg test:', np.min(x_test), ' / ', np.max(x_test), ' / ', np.mean(x_test))

        if norm_type == '255':
            x_train /= 255.
            x_test /= 255.
        elif norm_type == 'standard':
            mean = np.mean(x_train)
            std = np.std(x_train)
            x_train = (x_train - mean) / (std + 0.00001)
            x_test = (x_test - mean) / (std + 0.00001)
        elif norm_type == 'mean':
            mean = np.mean(x_train)
            x_train -= mean
            x_test -= mean
        elif norm_type == 'keras':
            x_train = imagenet_utils.preprocess_input(x_train, mode='tf')
            x_test = imagenet_utils.preprocess_input(x_test, mode='tf')

        print(' After norm...')
        print(' - Min / max / avg train:', np.min(x_train), ' / ', np.max(x_train), ' / ', np.mean(x_train))
        print(' - Min / max / avg test:', np.min(x_test), ' / ', np.max(x_test), ' / ', np.mean(x_test))

        datasets[i]['x_train'] = x_train
        datasets[i]['x_test'] = x_test


# -----------------------------------------------------------------------------
def load(config):
    if config.db == 'mnist':
        datasets = utilLoadMNIST.load_datasets(config.select, config.size, config.v)
    elif config.db == 'signs':
        datasets = utilLoadSigns.load_datasets(config.select, config.size, config.v)
    else:
        raise Exception('Unknowm dataset')

    input_shape, num_labels = __run_validations(datasets)

    # Label smooth
    if config.lsmooth:
        __smooth_labels(datasets)

    # Normalize
    __normalize(datasets, config.norm)

    # Shuffle
    for i in range(len(datasets)):
        datasets[i]['x_train'], datasets[i]['y_train'] = shuffle(datasets[i]['x_train'], datasets[i]['y_train'])
        datasets[i]['x_test'], datasets[i]['y_test'] = shuffle(datasets[i]['x_test'], datasets[i]['y_test'])

    # Truncate?
    if config.truncate:
        print('Truncate...')
        for i in range(len(datasets)):
            factor = 0.2 # 0.1
            new_len_tr = int( factor * len(datasets[i]['x_train']) )
            new_len_te = int( factor * len(datasets[i]['x_test']) )
            datasets[i]['x_train'] = datasets[i]['x_train'][:new_len_tr]
            datasets[i]['y_train'] = datasets[i]['y_train'][:new_len_tr]
            datasets[i]['x_test'] = datasets[i]['x_test'][:new_len_te]
            datasets[i]['y_test'] = datasets[i]['y_test'][:new_len_te]

    return datasets, input_shape, num_labels

