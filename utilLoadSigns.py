# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import cv2
import os
import warnings
import scipy.io
import pickle as pkl
import skimage.transform
import util
from sklearn.model_selection import StratifiedKFold
from keras.utils import np_utils
from tensorflow.examples.tutorials.mnist import input_data

warnings.filterwarnings('ignore')

ARRAY_BD_NAMES = ['gtsrb', 'syn_signs']

VERBOSE_NB_SHOW = 20


# ----------------------------------------------------------------------------
def __read_gtsrb_train_images(path, size):
    X = []
    Y = []
    for c in range(0, 43):       # loop over all 43 classes
        subpath = os.path.join(path, format(c, '05d'))
        for fname in util.list_files(subpath, 'ppm'):
            img = cv2.imread(fname, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (size, size), interpolation = cv2.INTER_CUBIC )
            #print(c, fname)
            #print(' - ', img.shape)
            #cv2.imshow("img", img)
            #cv2.waitKey(0)
            X.append( img )
            Y.append( c )

    return X, Y


# ----------------------------------------------------------------------------
def __read_gtsrb_test_images(imgpath, gt_csv_file, size):
    X = []
    Y = []
    csv_data = util.load_csv(gt_csv_file, sep=';', header=0)
    for i in range(len(csv_data)):
        img = cv2.imread(os.path.join(imgpath, csv_data[i, 0]), cv2.IMREAD_COLOR)
        img = cv2.resize(img, (size, size), interpolation = cv2.INTER_CUBIC )
        #print(c, fname)
        #print(' - ', img.shape)
        #cv2.imshow("img", img)
        #cv2.waitKey(0)
        X.append( img )
        Y.append( csv_data[i, 7] )
    return X, Y


# ----------------------------------------------------------------------------
# GTSRB dataset
# train image sizes- min:25 / max:243
# test image sizes - min 25  - max 266
def __load_dataset_gtsrb(img_size, verbose):
    basepath = 'datasets/GTSRB'

    x_train, y_train = __read_gtsrb_train_images( os.path.join(basepath, 'Final_Training/Images'), img_size )
    x_test, y_test = __read_gtsrb_test_images( os.path.join(basepath, 'Final_Test/Images'),
                                                                                                  os.path.join(basepath, 'Final_Test/GT-final_test.csv'),
                                                                                                  img_size )

    if verbose:
        print('y', np.min(y_train), np.max(y_train))
        for i in np.random.choice(len(x_train), VERBOSE_NB_SHOW, replace=False):
            print('Label:', y_train[i])
            #cv2.imwrite('IMGS/gtsrb'+str(i)+'.png',x_train[i])
            cv2.imshow("Img", x_train[i])
            cv2.waitKey(0)

    x_train = np.asarray(x_train)
    x_test = np.asarray(x_test)
    y_train = np_utils.to_categorical(y_train, num_classes=43)
    y_test = np_utils.to_categorical(y_test, num_classes=43)

    return x_train, y_train, x_test, y_test


# ----------------------------------------------------------------------------
# Syn Signs / Synthetic Signs dataset
# img size: 40x40x3
# synthetic_data/
# synthetic_data/train/   -- contains the images as PNGs
# synthetic_data/train_labelling.txt   -- ground truths
def __load_dataset_syn_signs(img_size, verbose):
    basepath = 'datasets/SYN_SIGNS/synthetic_data'
    size = -1   # 32
    X = []
    Y = []
    csv_filename = os.path.join(basepath, 'train_labelling.txt')
    csv_data = util.load_csv(csv_filename, sep=' ', header=None)

    for i in range(len(csv_data)):
        img = cv2.imread(os.path.join(basepath, csv_data[i, 0]), cv2.IMREAD_COLOR)
        if img_size != 40:
            img = cv2.resize(img, (img_size, img_size), interpolation = cv2.INTER_CUBIC )
        #print(csv_data[i, 0], csv_data[i, 1], img.shape)
        #cv2.imshow("img", img)
        #cv2.waitKey(0)
        X.append(img)
        Y.append(csv_data[i, 1])

    if verbose:
        print('y', np.min(Y), np.max(Y))
        for i in np.random.choice(len(X), VERBOSE_NB_SHOW, replace=False):
            print('Label:', Y[i])
            #cv2.imwrite('IMGS/syn_signs'+str(i)+'.png',X[i])
            cv2.imshow("Img", X[i])
            cv2.waitKey(0)

    skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
    train_index, test_index = next(skf.split(X, Y))

    X = np.asarray(X)
    Y = np_utils.to_categorical(Y, num_classes=43)

    x_train, x_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]

    return x_train, y_train, x_test, y_test


# ----------------------------------------------------------------------------
# Parameter:
# - selected_bds = array of bd names to load. None to load all them
def load_datasets(selected_bds, scale_to, verbose):
    return_array = []

    if selected_bds is None or len(selected_bds) == 0:
        selected_bds = ARRAY_BD_NAMES

    img_size = 40
    if scale_to > 0:
        img_size = scale_to

    for bd_name in selected_bds:
        assert bd_name in ARRAY_BD_NAMES
        print('Loading', bd_name, 'dataset...')
        x_train, y_train, x_test, y_test = globals()['__load_dataset_'+bd_name](img_size, verbose)
        print(' - X Train:', x_train.shape)
        print(' - Y Train:', y_train.shape)
        print(' - X Test:', x_test.shape)
        print(' - Y Test:', y_test.shape)

        return_array.append( {'name': bd_name,
                                                        'x_train': x_train, 'y_train': y_train,
                                                        'x_test': x_test, 'y_test': y_test
                                                    } )

    return return_array

