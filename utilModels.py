# -*- coding: utf-8 -*-
import abc
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras import backend as K


# ----------------------------------------------------------------------------
class AbstractModel(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, input_shape):
        self.input = Input(shape=input_shape, name="main_input")

    @abc.abstractmethod
    def get_model_features(self):
        pass

    @abc.abstractmethod
    def get_model_labels(self, input):
        pass

    @abc.abstractmethod
    def get_model_domains(self, input):
        pass


# ----------------------------------------------------------------------------
class ModelV0(AbstractModel):
    def get_model_features(self):
        net = Convolution2D(32, (3, 3), border_mode='valid', activation='relu')(self.input)
        net = Convolution2D(32, (3, 3), activation='relu')(net)
        net = MaxPooling2D(pool_size=(2, 2))(net)
        net = Dropout(0.5)(net)
        net = Flatten()(net)
        return net

    def get_model_labels(self, input):
        net = Dense(128, activation='relu', name='features_inc')(input)
        net = Dropout(0.5)(net)
        return net

    def get_model_domains(self, input):
        net = Dense(128, activation='relu')(input)
        net = Dropout(0.1)(net)
        return net


# ----------------------------------------------------------------------------
# v1 - mnist_model
class ModelV1(AbstractModel):
    def get_model_features(self):
        model = Conv2D(32, (3, 3), activation='relu')(self.input)
        model = Conv2D(64, (3, 3), activation='relu')(model)
        model = MaxPooling2D(pool_size=(2, 2))(model)
        model = Dropout(0.25)(model)
        model = Flatten()(model)
        return model

    def get_model_labels(self, input):
        model = Dense(128, activation='relu', name='features_inc')(input)
        model = Dropout(0.5)(model)
        return model

    def get_model_domains(self, input):
        model = Dense(128, activation='relu')(input)
        model = Dropout(0.5)(model)
        return model


# ----------------------------------------------------------------------------
# v2 - mensural_model
class ModelV2(AbstractModel):
    def get_model_features(self):
        model = Conv2D(32, (3, 3), padding='same', activation='relu')(self.input)
        model = Conv2D(64, (3, 3), padding='same', activation='relu')(model)
        model = MaxPooling2D(pool_size=(2, 2))(model)
        model = Dropout(0.25)(model)
        model = Conv2D(64, (3, 3), padding='same', activation='relu')(model)
        model = Conv2D(64, (3, 3), padding='same', activation='relu')(model)
        model = MaxPooling2D(pool_size=(2, 2))(model)
        model = Dropout(0.3)(model)
        model = Flatten()(model)
        return model

    def get_model_labels(self, input):
        model = Dense(128, activation='relu', name='features_inc')(input)
        model = Dropout(0.5)(model)
        return model

    def get_model_domains(self, input):
        model = Dense(128, activation='relu')(input)
        model = Dropout(0.5)(model)
        return model


# ----------------------------------------------------------------------------
# v3 - paper JMLR DANN used for SVHN (see Fig 4b)
class ModelV3(AbstractModel):
    def get_model_features(self):
        model = Conv2D(64, (5, 5), padding='same', activation='relu')(self.input)
        model = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(model)  # Original -> 3x3   (2, 2)
        model = Conv2D(64, (5, 5), padding='same', activation='relu')(model)
        model = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(model)  # Original -> 3x3   (2, 2)
        model = Conv2D(128, (5, 5), padding='same', activation='relu')(model)
        model = Flatten()(model)
        return model

    def get_model_labels(self, input):
        model = Dense(3072, activation='relu')(input)
        model = Dropout(0.5)(model)
        model = Dense(2048, activation='relu', name='features_inc')(model)
        model = Dropout(0.5)(model)
        return model

    def get_model_domains(self, input):
        model = Dense(1024, activation='relu')(input)
        model = Dropout(0.5)(model)
        model = Dense(1024, activation='relu')(model)
        model = Dropout(0.5)(model)
        return model


# ----------------------------------------------------------------------------
# v4 - paper JMLR DANN used for GTSRB (see Fig 4c)
class ModelV4(AbstractModel):
    def get_model_features(self):
        model = Conv2D(96, (5,5), padding='same', activation='relu')(self.input)
        model = MaxPooling2D(pool_size=(2, 2))(model)  # Original -> 3x3   (2, 2)
        model = Conv2D(144, (3,3), padding='same', activation='relu')(model)
        model = MaxPooling2D(pool_size=(2, 2))(model)  # Original -> 3x3   (2, 2)
        model = Conv2D(256, (5,5), padding='same', activation='relu')(model)
        model = MaxPooling2D(pool_size=(2, 2))(model)  # Original -> 3x3   (2, 2)
        model = Flatten()(model)
        return model

    def get_model_labels(self, input):
        model = Dense(512, activation='relu', name='features_inc')(input)
        model = Dropout(0.5)(model)
        return model

    def get_model_domains(self, input):
        model = Dense(1024, activation='relu')(input)
        model = Dropout(0.5)(model)
        model = Dense(1024, activation='relu')(model)
        model = Dropout(0.5)(model)
        return model


# ----------------------------------------------------------------------------
# v21 - MobileNetV2 - input_shape 224x224
# https://github.com/keras-team/keras-applications
class ModelV21(AbstractModel):
    def __init__(self, value):
        super().__init__()
        #keras.applications.mobilenet_v2.MobileNetV2(input_shape=None, alpha=1.0,
        # depth_multiplier=1, include_top=True, weights='imagenet', input_tensor=None,
        # pooling=None, classes=1000)
        self.base_model = applications.mobilenet_v2.MobileNetV2(weights='imagenet', include_top=False)
        self.input = base_model.input

    def get_model_features(self):
        model = self.base_model.output
        model = GlobalAveragePooling2D()(model)
        return model

    def get_model_labels(self, input):
        model = Dense(1024, activation='relu', name='features_inc')(input)
        model = Dropout(0.5)(model)
        return model

    def get_model_domains(self, input):
        model = Dense(1024, activation='relu')(input)
        model = Dropout(0.5)(model)
        return model


# ----------------------------------------------------------------------------
# v22 - NASNetLarge   - input_shape 331x331
class ModelV22(AbstractModel):
    def __init__(self, value):
        super().__init__()
        #keras.applications.nasnet.NASNetLarge(input_shape=None, include_top=True,
        #         weights='imagenet', input_tensor=None, pooling=None, classes=1000)
        # keras.applications.nasnet.NASNetMobile(input_shape=None, include_top=True,
        #         weights='imagenet', input_tensor=None, pooling=None, classes=1000)
        self.base_model = applications.nasnet.NASNetLarge(weights='imagenet', include_top=False)
        self.input = base_model.input

    def get_model_features(self):
        model = self.base_model.output
        model = GlobalAveragePooling2D()(model)
        return model

    def get_model_labels(self, input):
        model = Dense(1024, activation='relu', name='features_inc')(input)
        model = Dropout(0.5)(model)
        return model

    def get_model_domains(self, input):
        model = Dense(1024, activation='relu')(input)
        model = Dropout(0.5)(model)
        return model

