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

def default_generator(latent_dim, is_a_grayscale, nch=512, h=5, initial_size=4, final_size=512, div=[2,2,4,4,8,8,16], num_repeats=0):
    layer = InputLayer((None, latent_dim))
    layer = DenseLayer(layer, num_units=nch*initial_size*initial_size, nonlinearity=linear)
    layer = BatchNormLayer(layer)
    layer = ReshapeLayer(layer, (-1, nch, initial_size, initial_size))
    div = [nch/elem for elem in div]
    for n in div:
        for r in range(num_repeats+1):
            layer = Conv2DLayer(layer, num_filters=n, filter_size=h, pad='same', nonlinearity=linear)
            layer = BatchNormLayer(layer)
            layer = NonlinearityLayer(layer, nonlinearity=LeakyRectify(0.2))
            if dropout>0:
                layer = DropoutLayer(layer, p=0.1)
        layer = Upscale2DLayer(layer, scale_factor=2)
    layer = Conv2DLayer(layer, num_filters=1 if is_a_grayscale else 3, filter_size=h, pad='same', nonlinearity=sigmoid)
    return layer

"""
def model_discriminator(nch=512, h=5, div=[8,4,4,2,2,1,1], num_repeats=0, bn=False, pool_mode='max', reg=lambda: l1l2(l1=1e-7, l2=1e-7), nonlinearity='sigmoid' ):
    model = Sequential()
    n = nch / 8
    #div = [nch/8, nch/4, nch/4, nch/2, nch/2, nch/1, nch/1]
    div = [nch/elem for elem in div]
    for idx,n in enumerate(div):
        for r in range(num_repeats+1):
            if idx==0:
                model.add(Conv2D(n, h, strides=1, padding='same', kernel_regularizer=reg(),
                               input_shape=dim_ordering_shape((1, 512, 512))))
            else:
                model.add(Conv2D(n, h, strides=1, padding='same', kernel_regularizer=reg()))
            if bn:
                model.add(BatchNormalization(mode=0,axis=1))
            model.add(LeakyReLU(0.2))
        if pool_mode == 'max':
            model.add(MaxPooling2D(pool_size=(2, 2)))
        else:
            model.add(AveragePooling2D(pool_size=(2,2)))
    model.add(Conv2D(1, h, padding='same', kernel_regularizer=reg()))
    reduction_factor = nch // (2**len(div))
    model.add(AveragePooling2D(pool_size=(reduction_factor,reduction_factor), border_mode='valid')) #4x4
    model.add(Flatten())
    model.add(Activation(nonlinearity))
    return model
"""
def default_discriminator(is_a_grayscale, nch=512, h=5, div=[8,4,4,2,2,1,1], num_repeats=0, bn=False, pool_mode='max', nonlinearity='sigmoid'):
    layer = InputLayer((None, 1 if is_a_grayscale else 3, 512, 512))
    n = nch / 8
    #div = [nch/8, nch/4, nch/4, nch/2, nch/2, nch/1, nch/1]
    div = [nch/elem for elem in div]
    for idx,n in enumerate(div):
        for r in range(num_repeats+1):
            layer = Conv2DLayer(layer, num_filters=n, filter_size=h, pad='same', nonlinearity=linear)
            if bn:
                layer = BatchNormLayer(layer)
            layer = NonlinearityLayer(layer, nonlinearity=LeakyRectify(0.2))
        if pool_mode == 'max':
            layer = MaxPool2DLayer(layer, pool_size=2)
        else:
            layer = Pool2DLayer(layer, pool_size=2, mode='average_inc_pad')
    layer = Conv2DLayer(layer, num_filters=1, filter_size=h, pad='same')
    reduction_factor = nch // (2**len(div))
    layer = Pool2DLayer(layer, pool_size=(reduction_factor,reduction_factor), mode='average_inc_pad')
    #model.add(AveragePooling2D(pool_size=(reduction_factor,reduction_factor), border_mode='valid')) #4x4
    #model.add(Flatten())
    layer = ReshapeLayer(layer, (-1, 1))
    layer = NonlinearityLayer(layer, nonlinearity)
    # ------
    return layer


if __name__ == '__main__':
    #l_out = default_generator(True, 256)
    #for layer in get_all_layers(l_out):
    #    print layer, layer.output_shape

    #l_out = default_discriminator()
    #for layer in get_all_layers(l_out):
    #    print layer, layer.output_shape

    pass
    


"""
def model_generator(latent_dim, nch=512, h=5, initial_size=4, final_size=512, div=[2,2,4,4,8,8,16], num_repeats=0, reg=lambda: l1l2(l1=1e-7, l2=1e-7), dropout=0.):
    # e.g. for 512x512 generation, if we start at 4x4,
    # we are only allowed 7 upsampling div to take us to
    # 512px
    assert initial_size * (2**len(div)) == final_size
    model = Sequential()
    model.add(Dense(nch * initial_size * initial_size, input_dim=latent_dim, kernel_regularizer=reg()))
    model.add(BatchNormalization(axis=1)) ##?
    model.add(Reshape(dim_ordering_shape((nch, initial_size, initial_size))))
    div = [nch/elem for elem in div]
    print div
    #div = [nch/2, nch/2, nch/4, nch/4, nch/8, nch/8, nch/16]
    for n in div:
        for r in range(num_repeats+1):
            model.add(Conv2D(n, h, padding='same', kernel_regularizer=reg()))
            model.add(BatchNormalization(mode=0, axis=1))
            model.add(LeakyReLU(0.2))
            if dropout > 0:
                model.add(Dropout(0.1))
        model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(1, h, padding='same', kernel_regularizer=reg()))
    model.add(Activation('sigmoid'))
    return model
"""
