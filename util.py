# -*- coding: utf-8 -*-
from __future__ import print_function
import sys, os, re
import pandas as pd
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import tensorflow as tf


#------------------------------------------------------------------------------
def init():
    np.set_printoptions(threshold=sys.maxsize)
    sys.setrecursionlimit(40000)
    random.seed(42)                             # For reproducibility
    np.random.seed(42)
    tf.set_random_seed(42)


# ----------------------------------------------------------------------------
def pause():
    programPause = raw_input("Press the <ENTER> key to continue...")


# ----------------------------------------------------------------------------
def mkdirp(directory):
    if not os.path.isdir(directory):
        os.makedirs(directory)


# ----------------------------------------------------------------------------
# Return the list of files in folder
def list_dirs(directory):
    return [os.path.join(directory, f) for f in os.listdir(directory)
            if os.path.isdir(os.path.join(directory, f))]


# ----------------------------------------------------------------------------
# Return the list of files in folder
# ext param is optional. For example: 'jpg' or 'jpg|jpeg|bmp|png'
def list_files(directory, ext=None):
    return [os.path.join(directory, f) for f in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, f)) and ( ext==None or re.match('([\w_-]+\.(?:' + ext + '))', f) )]


# ----------------------------------------------------------------------------
# http://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html
def load_csv(path, sep=',', header=None):
	df = pd.read_csv(path, sep=sep, header=header)
	return df.values


#------------------------------------------------------------------------------
def l2norm(X):
    norm = 0
    for i in range(len(X)):
        if X[i] < 0:
            X[i] = 0
        else:
            norm += X[i] * X[i]
    if norm != 0:
        norm = math.sqrt(norm)
        X /= norm


# ----------------------------------------------------------------------------
# Convert a matrix of one-hot row-vector labels into smoothed versions.
# Label smoothing ref: https://arxiv.org/pdf/1812.01187v2.pdf
# Arguments
#        y: matrix of one-hot row-vector labels to be smoothed
#        smooth_factor: label smoothing factor (between 0 and 1)
# Returns
#        A matrix of smoothed labels.
def smooth_labels(y, smooth_factor=.1):
    assert len(y.shape) == 2
    assert 0 <= smooth_factor <= 1, 'Invalid label smoothing factor: ' + str(smooth_factor)
    y *= 1. - smooth_factor
    y += smooth_factor / y.shape[1]
    return y


# ----------------------------------------------------------------------------
class MyEarlyStopping():
    def __init__(self, patience=0, monitor='min'):
        assert patience > 1
        self.patience = patience
        self.monitor_op = np.less if monitor=='min' else np.greater
        self.best = np.Inf if monitor=='min' else -np.Inf
        self.wait = 0

    def check_stop(self, loss):
        if self.monitor_op(loss, self.best):
            self.best = loss
            self.wait = 0
        else:
            if self.wait >= self.patience:
                print(' - Early stopping!')
                return True
            self.wait += 1
        return False


# ----------------------------------------------------------------------------
def show_histogram(values):
    plt.hist(values)
    plt.xlabel("Prob")
    plt.ylabel('Frequency');
    plt.show()


# -----------------------------------------------------------------------------
def plot_bar(_prob, sort=True):
    prob = _prob.copy()
    if sort:
        prob.sort()
    plt.bar(np.arange(len(prob)), prob)
    plt.show()


# ----------------------------------------------------------------------------
def imshow_grid(images, shape=[2, 8]):
    fig = plt.figure(1)
    grid = ImageGrid(fig, 111, nrows_ncols=shape, axes_pad=0.05)
    size = shape[0] * shape[1]
    for i in range(size):
        grid[i].axis('off')
        grid[i].imshow(images[i])
    plt.show()


# ----------------------------------------------------------------------------
def get_categorical_map_from_list(Y):
    categories = set(Y)
    category_map = dict([(char,i) for i, char in enumerate(categories)])
    return category_map

