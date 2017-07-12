import theano
from theano import tensor as T
import lasagne
from lasagne.layers import *
from lasagne.nonlinearities import *
from lasagne.init import *
from lasagne.updates import *
from lasagne.objectives import *
from keras.preprocessing.image import ImageDataGenerator
import os
import sys

class BilinearUpsample2DLayer(Layer):
    def __init__(self, incoming, factor, **kwargs):
        super(BilinearUpsample2DLayer, self).__init__(incoming, **kwargs)
        self.factor = factor

    def get_output_shape_for(self, input_shape):
        return input_shape[0:2] + (input_shape[2]*self.factor, input_shape[3]*self.factor)

    def get_output_for(self, input, **kwargs):
        return theano.tensor.nnet.abstract_conv.bilinear_upsampling(
            input, 
            self.factor, 
            batch_size=self.input_shape[0],
            num_input_channels=self.input_shape[1])

