# -*- coding: utf-8 -*-
"""
GRADIENT REVERSAL LAYER
Original version:
https://github.com/michetonu/gradient_reversal_keras_tf
Modified version:
- Fix errors on lambda
- Encapsulate code
- Add methods to modify lambda during training
"""
from keras.engine import Layer
from keras import backend as K
import uuid
import tensorflow as tf

# ----------------------------------------------------------------------------
class GradientReversal(Layer):
    '''Flip the sign of gradient during training.'''

    # --------------------------------------
    def __init__(self, hp_lambda, **kwargs):
        super(GradientReversal, self).__init__(**kwargs)
        self.supports_masking = False
        #self.hp_lambda = hp_lambda
        self.hp_lambda = K.variable(hp_lambda, dtype='float32', name='hp_lambda')

    # --------------------------------------
    def build(self, input_shape):
        self.trainable_weights = []

    # --------------------------------------
    def reverse_gradient(self, X):
        '''Flips the sign of the incoming gradient during training.'''
        """try:
            self.num_calls += 1
        except AttributeError:
            self.num_calls = 1"""

        grad_name = "GradientReversal%d" % uuid.uuid4()           # self.num_calls

        @tf.RegisterGradient(grad_name)
        def _flip_gradients(op, grad):
            return [tf.negative(grad) * self.hp_lambda]

        g = K.get_session().graph
        with g.gradient_override_map({'Identity': grad_name}):
            y = tf.identity(X)

        return y

    # --------------------------------------
    def call(self, x, mask=None):
        return self.reverse_gradient(x)

    # --------------------------------------
    def get_output_shape_for(self, input_shape):
        return input_shape

    # --------------------------------------
    def set_hp_lambda(self,hp_lambda):
        #self.hp_lambda = hp_lambda
        K.set_value(self.hp_lambda, hp_lambda)

    # --------------------------------------
    def increment_hp_lambda_by(self,increment):
        new_value = float(K.get_value(self.hp_lambda)) +  increment
        K.set_value(self.hp_lambda, new_value)

    # --------------------------------------
    def get_hp_lambda(self):
        return float(K.get_value(self.hp_lambda))

    # --------------------------------------
    def get_config(self):
        config = {}
        base_config = super(GradientReversal, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

