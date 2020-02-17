# -*- coding: utf-8 -*-
import utilModels
from keras.models import Model
from keras.layers import Dense
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf


# ----------------------------------------------------------------------------
def get_cnn_weights_filename(weights_folder, dataset_name, config):
    return '{}{}/weights_cnn_model_v{}_for_{}_e{}_b{}{}.npy'.format(
                            weights_folder, ('/truncated' if config.truncate else ''),
                            str(config.model), dataset_name,
                            str(config.epochs), str(config.batch),
                            ('_aug' if config.aug else ''))


# -------------------------------------------------------------------------
def label_smoothing_loss(y_true, y_pred):
    return tf.losses.sigmoid_cross_entropy(y_true, y_pred, label_smoothing=0.01)


# -------------------------------------------------------------------------
'''Create the source or label model separately'''
def build_source_model(model_number, input_shape, nb_classes, config):
    auxModel = getattr(utilModels, "ModelV" + str(model_number))(input_shape)

    net = auxModel.get_model_features()
    net = auxModel.get_model_labels( net )
    net = Dense(nb_classes, activation='softmax', name='classifier_output')(net)

    model = Model(input=auxModel.input, output=net)

    opt = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(loss={'classifier_output': 'categorical_crossentropy'},
                                               optimizer=opt, metrics=['accuracy'])
    if config.v == True:
        print(model.summary())

    return model


# ----------------------------------------------------------------------------
def train_cnn_on_one_dataset(model,  source_x_train, source_y_train,
                                                                 source_x_test, source_y_test, weights_filename, config):
    early_stopping = EarlyStopping(monitor='loss', patience=15)

    validation_data=None
    if len(source_x_test) > 0:
        validation_data =(source_x_test, source_y_test)

    if config.aug == True:
        print('Fit CNN using data augmentation...')
        aug = ImageDataGenerator(
                                        rotation_range=1,
                                        vertical_flip=False,
                                        horizontal_flip=False,
                                        width_shift_range=0.05,
                                        height_shift_range=0.05,
                                        shear_range=0.0,
                                        zoom_range=0.05,
                                        fill_mode="nearest")
        model.fit_generator(  aug.flow(source_x_train, source_y_train, batch_size=config.batch),
                                        steps_per_epoch=len(source_x_train) // config.batch,
                                        epochs=config.epochs,
                                        validation_data=validation_data,
                                        verbose=2,
                                        callbacks=[early_stopping])
    else:
        print('Fit CNN...')
        model.fit(source_x_train, source_y_train,
                            batch_size=config.batch,
                            epochs=config.epochs,
                            verbose=2,
                            shuffle=True,
                            validation_data=validation_data,
                            callbacks=[early_stopping])

    model.save(weights_filename)

    return model


# ----------------------------------------------------------------------------
def train_cnn(datasets, input_shape, num_labels, weights_folder, config):
    for i in range(len(datasets)):
        if config.from_db is not None and config.from_db != datasets[i]['name']:
            continue

        model = build_source_model(config.model, input_shape, num_labels, config)

        print('BD: {} \tx_train:{}\ty_train:{}\tx_test:{}\ty_test:{}'.format(
                    datasets[i]['name'],
                    datasets[i]['x_train'].shape, datasets[i]['y_train'].shape,
                    datasets[i]['x_test'].shape, datasets[i]['y_test'].shape))

        weights_filename = get_cnn_weights_filename(weights_folder, datasets[i]['name'], config)

        if config.load == False:
            model = train_cnn_on_one_dataset(model,
                                                                datasets[i]['x_train'], datasets[i]['y_train'],
                                                                datasets[i]['x_test'], datasets[i]['y_test'],
                                                                weights_filename, config)
        else:
            model = load_model(weights_filename)

        # Final evaluation
        print(80*'-')
        print('FINAL VALIDATION:')
        source_loss, source_acc = model.evaluate(datasets[i]['x_test'], datasets[i]['y_test'], verbose=0)
        print(' - Source test set "{}" accuracy: {:.4f}'.format(datasets[i]['name'], source_acc))

        for j in range(len(datasets)):
            if i == j or (config.to_db is not None and config.to_db != datasets[j]['name']):
                continue
            target_loss, target_acc = model.evaluate(datasets[j]['x_test'], datasets[j]['y_test'], verbose=0)
            print(' - Target test set "{}" accuracy: {:.4f}'.format(datasets[j]['name'], target_acc))

