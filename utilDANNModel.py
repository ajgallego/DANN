# -*- coding: utf-8 -*-
import utilModels
import numpy as np
from utilGradientReversal import GradientReversal
from keras import applications
from keras import optimizers
from keras.models import Model
from keras.layers import Dense, Lambda
from keras import backend as K

class DANNModel(object):
    # -------------------------------------------------------------------------
    def __init__(self, model_number, input_shape, nb_classes, batch_size, grl='auto', summary=False):
        self.learning_phase = K.variable(1)
        self.model_number = model_number
        self.input_shape = input_shape
        self.nb_classes = nb_classes
        self.batch_size = batch_size

        self.opt = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

        self.clsModel = getattr(utilModels, "ModelV" + str(model_number))(input_shape)

        self.dann_model, self.label_model, self.tsne_model = self.__build_dann_model()

        self.compile()

    # -------------------------------------------------------------------------
    def load(self, filename):
        weight = np.load(filename, allow_pickle=True)
        self.dann_model.set_weights(weight)

    # -------------------------------------------------------------------------
    def save(self, filename):
        np.save(filename, self.dann_model.get_weights())

    # -------------------------------------------------------------------------
    def compile(self):
        self.dann_model.compile(loss={'classifier_output': 'categorical_crossentropy',
                                                                            'domain_output': 'categorical_crossentropy'},
                                                               loss_weights={'classifier_output': 0.5, 'domain_output': 1.0},
                                                               optimizer=self.opt,
                                                               metrics=['accuracy'])

        self.label_model.compile(loss='categorical_crossentropy',
                                                                 optimizer=self.opt,
                                                                 metrics=['accuracy'])

        self.tsne_model.compile(loss='categorical_crossentropy',
                                                                optimizer=self.opt,
                                                                metrics=['accuracy'])


    # -------------------------------------------------------------------------
    def __build_dann_model(self):
        branch_features = self.clsModel.get_model_features()

        # Build domain model...
        self.grl_layer = GradientReversal(1.0)
        branch_domain = self.grl_layer(branch_features)
        branch_domain = self.clsModel.get_model_domains(branch_domain)
        branch_domain = Dense(2, activation='softmax', name='domain_output')(branch_domain)

        # Build label model...
        # When building DANN model, route first half of batch (source examples)
        # to domain classifier, and route full batch (half source, half target)
        # to the domain classifier.
        branch_label = Lambda(lambda x: K.switch(K.learning_phase(),
                                                                                                 K.concatenate([x[:int(self.batch_size//2)],
                                                                                                                                   x[:int(self.batch_size//2)]], axis=0),
                                                                                                 x),
                                                        output_shape=lambda x: x[0:])(branch_features)

        # Build label model...
        branch_label = self.clsModel.get_model_labels(branch_label)
        branch_label = Dense(self.nb_classes, activation='softmax', name='classifier_output')(branch_label)

        # Create models...
        dann_model = Model(input=self.clsModel.input, output=[branch_domain, branch_label])
        label_model = Model(input=self.clsModel.input, output=branch_label)
        tsne_model = Model(input=self.clsModel.input, output=branch_features)

        if config.v == True:
            print(dann_model.summary())

        return dann_model, label_model, tsne_model
