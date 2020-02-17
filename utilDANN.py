# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import gc
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.models import load_model
import tensorflow as tf
import util


# ----------------------------------------------------------------------------
def get_dann_weights_filename(folder, from_dataset, to_dataset, config):
    return '{}{}/weights_dann_model_v{}_from_{}_to_{}_e{}_b{}.npy'.format(
                            folder,
                            ('/truncated' if config.truncate else ''),
                            str(config.model), from_dataset, to_dataset,
                            str(config.epochs), str(config.batch))


# ----------------------------------------------------------------------------
def batch_generator(x_data, y_data=None, batch_size=1, shuffle_data=True):
    len_data = len(x_data)
    index_arr = np.arange(len_data)
    if shuffle_data:
        np.random.shuffle(index_arr)

    start = 0
    while len_data > start + batch_size:
        batch_ids = index_arr[start:start + batch_size]
        start += batch_size
        if y_data is not None:
            x_batch = x_data[batch_ids]
            y_batch = y_data[batch_ids]
            yield x_batch, y_batch
        else:
            x_batch = x_data[batch_ids]
            yield x_batch


# ----------------------------------------------------------------------------
def train_dann_batch(dann_model, src_generator, target_genenerator, target_x_train, batch_size):
    for batchXs, batchYs in src_generator:
        try:
            batchXd = next(target_genenerator)
        except: # Restart...
            target_genenerator = batch_generator(target_x_train, None, batch_size=batch_size // 2)
            batchXd = next(target_genenerator)

        # Combine the labeled and unlabeled data along with the discriminative results
        combined_batchX = np.concatenate((batchXs, batchXd))
        batch2Ys = np.concatenate((batchYs, batchYs))
        batchYd = np.concatenate((np.tile([0, 1], [batch_size // 2, 1]),
                                                                    np.tile([1, 0], [batch_size // 2, 1])))
        #print(combined_batchX.shape, batch2Ys.shape, batchYd.shape)

        result = dann_model.train_on_batch(combined_batchX,
                                                                                     {'classifier_output': batch2Ys,
                                                                                       'domain_output':batchYd})

    #print(dann_builder.dann_model.metrics_names)
    return result


# ----------------------------------------------------------------------------
def train_dann(dann_builder, source_x_train, source_y_train, target_x_train, target_y_train,
                                 nb_epochs, batch_size, weights_filename, initial_hp_lambda=0.01,
                                 target_x_test = None, target_y_test = None):
    print('Training DANN model')
    best_label_acc = 0
    target_genenerator = batch_generator(target_x_train, None, batch_size=batch_size // 2)
    dann_builder.grl_layer.set_hp_lambda(initial_hp_lambda)

    for e in range(nb_epochs):
        src_generator = batch_generator(source_x_train, source_y_train, batch_size=batch_size // 2)

        # Update learning rates
        lr = float(K.get_value(dann_builder.opt.lr))* (1. / (1. + float(K.get_value(dann_builder.opt.decay)) * float(K.get_value(dann_builder.opt.iterations)) ))
        print(' - Lr:', lr, ' / Lambda:', dann_builder.grl_layer.get_hp_lambda())

        dann_builder.grl_layer.increment_hp_lambda_by(1e-4)

        # Train batch
        loss, domain_loss, label_loss, domain_acc, label_acc = train_dann_batch(
                                            dann_builder.dann_model, src_generator, target_genenerator, target_x_train, batch_size )

        saved = ""
        if best_label_acc <= label_acc:
            best_label_acc = label_acc
            dann_builder.save(weights_filename)
            saved = "SAVED"

        if target_genenerator is not None:
            if target_x_test is not None:
                target_loss, target_acc = dann_builder.label_model.evaluate(target_x_test, target_y_test, batch_size=32, verbose=0)
            else:
                target_loss, target_acc = dann_builder.label_model.evaluate(target_x_train, target_y_train, batch_size=32, verbose=0)
        else:
            domain_loss, domain_acc = -1, -1
            target_loss, target_acc = -1, -1

        print("Epoch [{}/{}]: source label loss = {:.4f}, acc = {:.4f} | domain loss = {:.4f}, acc = {:.4f} | target label loss = {:.4f}, acc = {:.4f} | {}".format(
                            e+1, nb_epochs, label_loss, label_acc, domain_loss, domain_acc, target_loss, target_acc, saved))

        gc.collect()


