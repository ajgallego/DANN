# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import cv2
import os
import scipy.io
import pickle as pkl
import skimage.transform
from sklearn.datasets import load_svmlight_file
from keras.utils import np_utils
from tensorflow.examples.tutorials.mnist import input_data
import warnings
warnings.filterwarnings('ignore')

ARRAY_BD_NAMES = ['mnist', 'mnist_m', 'svhn', 'syn_numbers']

VERBOSE_NB_SHOW = 20


# ----------------------------------------------------------------------------
def __resize_array_images(array_images, size):
    new_array = []
    for i in range(len(array_images)):
        img = cv2.resize( array_images[i], (size, size), interpolation = cv2.INTER_CUBIC )
        new_array.append( img )
    return np.array(new_array)


# ----------------------------------------------------------------------------
# MNIST
def __load_dataset_mnist(img_size, verbose):
    path = 'datasets/MNIST_data'
    mnist_tf = input_data.read_data_sets(path, one_hot=True)

    x_train = (mnist_tf.train.images * 255).reshape(55000, 28, 28, 1).astype(np.uint8)
    x_train = np.concatenate([x_train, x_train, x_train], 3)

    x_test = (mnist_tf.test.images * 255).reshape(10000, 28, 28, 1).astype(np.uint8)
    x_test = np.concatenate([x_test, x_test, x_test], 3)

    y_train = mnist_tf.train.labels
    y_test = mnist_tf.test.labels

    x_train = np.asarray(x_train)
    x_test = np.asarray(x_test)

    if img_size != 28:
        x_train = __resize_array_images(x_train, img_size)
        x_test = __resize_array_images(x_test, img_size)

    if verbose:
        print('y', np.min(y_train), np.max(y_train))
        for i in np.random.choice(len(x_train), VERBOSE_NB_SHOW, replace=False):
            print('Label:', y_train[i])
            cv2.imshow("mat", x_train[i])
            cv2.waitKey(0)

    return x_train, y_train, x_test, y_test


# ----------------------------------------------------------------------------
# MNIST-M
def __load_dataset_mnist_m(img_size, verbose):
    path = 'datasets/dataset_target_mnist_data.pkl'
    mnistm_mnist = pkl.load(open(path, 'rb'), encoding='latin1')

    x_train = mnistm_mnist['x_train']
    y_train = mnistm_mnist['y_train']
    x_test = mnistm_mnist['x_test']
    y_test = mnistm_mnist['y_test']

    if img_size != 28:
        x_train = __resize_array_images(x_train, img_size)
        x_test = __resize_array_images(x_test, img_size)

    if verbose:
        print('y', np.min(y_train), np.max(y_train))
        for i in np.random.choice(len(x_train), VERBOSE_NB_SHOW, replace=False):
            print('Label:', y_train[i])
            cv2.imshow("mat", x_train[i])
            cv2.waitKey(0)

    x_train = np.asarray(x_train)
    x_test = np.asarray(x_test)

    return x_train, y_train, x_test, y_test


# ----------------------------------------------------------------------------
# SVHN dataset
# Loading the .mat files creates 2 variables: X which is a 4-D matrix containing the
# images, and y which is a vector of class labels. To access the images, X(:,:,:,i)
# gives the i-th 32-by-32 RGB image, with class label y(i).
def __load_dataset_svhn(img_size, verbose):
    path = 'datasets/SVHN'
    mat_train = scipy.io.loadmat(os.path.join(path, 'train_32x32.mat'))
    mat_test = scipy.io.loadmat(os.path.join(path, 'test_32x32.mat'))
    x_train = mat_train['X']
    y_train = mat_train['y'].flatten()
    x_test = mat_test['X']
    y_test = mat_test['y'].flatten()

    x_train = np.rollaxis(x_train, 3, 0)
    x_test = np.rollaxis(x_test, 3, 0)

    x_train = __resize_array_images(x_train, img_size)
    x_test = __resize_array_images(x_test, img_size)

    y_train[y_train == 10] = 0
    y_test[y_test == 10] = 0

    if verbose:
        print('y', np.min(y_train), np.max(y_train))
        for i in np.random.choice(len(x_train), VERBOSE_NB_SHOW, replace=False):
            print('Label:', y_train[i])
            cv2.imshow("mat", x_train[i])
            cv2.waitKey(0)

    x_train = np.asarray(x_train)
    x_test = np.asarray(x_test)

    y_train = np_utils.to_categorical(y_train, num_classes=10)
    y_test = np_utils.to_categorical(y_test, num_classes=10)

    return x_train, y_train, x_test, y_test


# ----------------------------------------------------------------------------
# Synth numbers or Synth
# Loading the .mat files creates 2 variables: X which is a 4-D matrix containing the
# images, and y which is a vector of class labels. To access the images, X(:,:,:,i)
# gives the i-th 32-by-32 RGB image, with class label y(i).
def __load_dataset_syn_numbers(img_size, verbose):
    path = 'datasets/SynthDigits'
    mat_train = scipy.io.loadmat(os.path.join(path, 'synth_train_32x32.mat'))
    mat_test = scipy.io.loadmat(os.path.join(path, 'synth_test_32x32.mat'))
    x_train = mat_train['X']
    y_train = mat_train['y'].flatten()
    x_test = mat_test['X']
    y_test = mat_test['y'].flatten()

    x_train = np.rollaxis(x_train, 3, 0)
    x_test = np.rollaxis(x_test, 3, 0)

    x_train = __resize_array_images(x_train, img_size)
    x_test = __resize_array_images(x_test, img_size)

    if verbose:
        print('y', np.min(y_train), np.max(y_train))
        for i in np.random.choice(len(x_train), VERBOSE_NB_SHOW, replace=False):
            print('Label:', y_train[i])
            cv2.imshow("mat", x_train[i]) #mat['X'][:, :, :, 0])
            cv2.waitKey(0)

    x_train = np.asarray(x_train)
    x_test = np.asarray(x_test)

    #print(np.min(y_train), np.max(y_train), np.min(y_test), np.max(y_test))
    #y_train[y_train == 10] = 0
    #y_test[y_test == 10] = 0
    y_train = np_utils.to_categorical(y_train, num_classes=10)
    y_test = np_utils.to_categorical(y_test, num_classes=10)

    return x_train, y_train, x_test, y_test


# ----------------------------------------------------------------------------
# Parameter:
# - selected_bds = array of bd names to load. None to load all them
def load_datasets(selected_bds, scale_to, verbose):
    return_array = []

    if selected_bds is None or len(selected_bds) == 0:
        selected_bds = ARRAY_BD_NAMES

    img_size = 28
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

