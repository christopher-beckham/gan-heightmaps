import os

import keras
from keras import backend as K
from keras import objectives
from keras.layers import Input, merge
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Convolution2D, Deconvolution2D
from keras.layers.core import Activation, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import Adam


KERAS_2 = keras.__version__[0] == '2'
try:
    # keras 2 imports
    from keras.layers.convolutional import Conv2DTranspose
    from keras.layers.merge import Concatenate
except ImportError:
    print("Keras 2 layers could not be imported defaulting to keras1")
    KERAS_2 = False

K.set_image_dim_ordering('th')


def concatenate_layers(inputs, concat_axis, mode='concat'):
    assert mode in ['concat','add']
    if mode == 'concat':
        return Concatenate(axis=concat_axis)(inputs)
    else:
        return Add()(inputs)

def Convolution(f, k=3, s=2, border_mode='same', **kwargs):
    """Convenience method for Convolutions."""
    return Convolution2D(f,
                         kernel_size=(k, k),
                         padding=border_mode,
                         strides=(s, s),
                         **kwargs)

def Deconvolution(f, output_shape, k=2, s=2, **kwargs):
    """Convenience method for Transposed Convolutions."""
    return Conv2DTranspose(f,
                           kernel_size=(k, k),
                           strides=(s, s),
                           data_format=K.image_data_format(),
                           **kwargs)

    
def BatchNorm(mode=2, axis=1, **kwargs):
    """Convenience method for BatchNormalization layers."""
    if KERAS_2:
        return BatchNormalization(axis=axis, **kwargs)
    else:
        return BatchNormalization(mode=2,axis=axis, **kwargs)
    #return lambda x: x


def g_unet(in_ch, out_ch, nf, batch_size=1, is_grayscale=False, num_padded_conv=0, concat_mode='concat', name='unet'):
    # type: (int, int, int, int, bool, str) -> keras.models.Model
    """Define a U-Net.

    Input has shape in_ch x 512 x 512
    Parameters:
    - in_ch: the number of input channels;
    - out_ch: the number of output channels;
    - nf: the number of filters of the first layer;
    - is_binary: if is_binary is true, the last layer is followed by a sigmoid
    - num_padded_conv: how many output shape preserving conv-batchnorm-relus should we add
      between the strided convolutions?
    activation function, otherwise, a tanh is used.
    """
    merge_params = {
        'mode': 'concat',
        'concat_axis': 1
    }
    if K.image_dim_ordering() == 'th':
        print('TheanoShapedU-NET')
        i = Input(shape=(in_ch, 512, 512))

        def get_deconv_shape(samples, channels, x_dim, y_dim):
            return samples, channels, x_dim, y_dim

    elif K.image_dim_ordering() == 'tf':
        i = Input(shape=(512, 512, in_ch))
        print('TensorflowShapedU-NET')

        def get_deconv_shape(samples, channels, x_dim, y_dim):
            return samples, x_dim, y_dim, channels

        merge_params['concat_axis'] = 3
    else:
        raise ValueError(
            'Keras dimension ordering not supported: {}'.format(
                K.image_dim_ordering()))

    def padded_conv(nf, x):
        for z in range(num_padded_conv):
            x = Convolution(nf,s=1,k=1)(x)
            x = BatchNorm()(x)
            x = LeakyReLU(0.2)(x)
        return x


    # in_ch x 512 x 512
    conv1 = Convolution(nf)(i)
    conv1 = BatchNorm()(conv1)
    x = LeakyReLU(0.2)(conv1)
    for r in range(num_padded_conv):
        x = padded_conv(nf, x)
    # nf x 256 x 256

    conv2 = Convolution(nf * 2)(x)
    conv2 = BatchNorm()(conv2)
    x = LeakyReLU(0.2)(conv2)
    for r in range(num_padded_conv):
        x = padded_conv(nf*2, x)
    # nf*2 x 128 x 128

    conv3 = Convolution(nf * 4)(x)
    conv3 = BatchNorm()(conv3)
    x = LeakyReLU(0.2)(conv3)
    for r in range(num_padded_conv):
        x = padded_conv(nf*4, x)
    # nf*4 x 64 x 64

    conv4 = Convolution(nf * 8)(x)
    conv4 = BatchNorm()(conv4)
    x = LeakyReLU(0.2)(conv4)
    for r in range(num_padded_conv):
        x = padded_conv(nf*8, x)
    # nf*8 x 32 x 32

    conv5 = Convolution(nf * 8)(x)
    conv5 = BatchNorm()(conv5)
    x = LeakyReLU(0.2)(conv5)
    for r in range(num_padded_conv):
        x = padded_conv(nf*8, x)
    # nf*8 x 16 x 16

    conv6 = Convolution(nf * 8)(x)
    conv6 = BatchNorm()(conv6)
    x = LeakyReLU(0.2)(conv6)
    for r in range(num_padded_conv):
        x = padded_conv(nf*8, x)
    # nf*8 x 8 x 8

    conv7 = Convolution(nf * 8)(x)
    conv7 = BatchNorm()(conv7)
    x = LeakyReLU(0.2)(conv7)
    for r in range(num_padded_conv):
        x = padded_conv(nf*8, x)
    # nf*8 x 4 x 4

    conv8 = Convolution(nf * 8)(x)
    conv8 = BatchNorm()(conv8)
    x = LeakyReLU(0.2)(conv8)
    for r in range(num_padded_conv):
        x = padded_conv(nf*8, x)
    # nf*8 x 2 x 2

    conv9 = Convolution(nf * 8, k=2, s=1, border_mode='valid')(x)
    conv9 = BatchNorm()(conv9)
    x = LeakyReLU(0.2)(conv9)
    for r in range(num_padded_conv):
        x = padded_conv(nf*8, x)
    # nf*8 x 1 x 1

    dconv1 = Deconvolution(nf * 8,
                           get_deconv_shape(batch_size, nf * 8, 2, 2),
                           k=2, s=1)(x)
    dconv1 = BatchNorm()(dconv1)
    x = concatenate_layers([dconv1, conv8], **merge_params)
    x = LeakyReLU(0.2)(x)
    for r in range(num_padded_conv):
        x = padded_conv(nf*8, x)
    # nf*(8 + 8) x 2 x 2

    dconv2 = Deconvolution(nf * 8,
                           get_deconv_shape(batch_size, nf * 8, 4, 4))(x)
    dconv2 = BatchNorm()(dconv2)
    x = concatenate_layers([dconv2, conv7], **merge_params)
    x = LeakyReLU(0.2)(x)
    for r in range(num_padded_conv):
        x = padded_conv(nf*8, x)
    # nf*(8 + 8) x 4 x 4

    dconv3 = Deconvolution(nf * 8,
                           get_deconv_shape(batch_size, nf * 8, 8, 8))(x)
    dconv3 = BatchNorm()(dconv3)
    x = concatenate_layers([dconv3, conv6], **merge_params)
    x = LeakyReLU(0.2)(x)
    for r in range(num_padded_conv):
        x = padded_conv(nf*8, x)
    # nf*(8 + 8) x 8 x 8

    dconv4 = Deconvolution(nf * 8,
                           get_deconv_shape(batch_size, nf * 8, 16, 16))(x)
    dconv4 = BatchNorm()(dconv4)
    x = concatenate_layers([dconv4, conv5], **merge_params)
    x = LeakyReLU(0.2)(x)
    for r in range(num_padded_conv):
        x = padded_conv(nf*8, x)
    # nf*(8 + 8) x 16 x 16

    dconv5 = Deconvolution(nf * 8,
                           get_deconv_shape(batch_size, nf * 8, 32, 32))(x)
    dconv5 = BatchNorm()(dconv5)
    x = concatenate_layers([dconv5, conv4], **merge_params)
    x = LeakyReLU(0.2)(x)
    for r in range(num_padded_conv):
        x = padded_conv(nf*8, x)
    # nf*(8 + 8) x 32 x 32

    dconv6 = Deconvolution(nf * 4,
                           get_deconv_shape(batch_size, nf * 4, 64, 64))(x)
    dconv6 = BatchNorm()(dconv6)
    x = concatenate_layers([dconv6, conv3], **merge_params)
    x = LeakyReLU(0.2)(x)
    for r in range(num_padded_conv):
        x = padded_conv(nf*4, x)
    # nf*(4 + 4) x 64 x 64

    dconv7 = Deconvolution(nf * 2,
                           get_deconv_shape(batch_size, nf * 2, 128, 128))(x)
    dconv7 = BatchNorm()(dconv7)
    x = concatenate_layers([dconv7, conv2], **merge_params)
    x = LeakyReLU(0.2)(x)
    for r in range(num_padded_conv):
        x = padded_conv(nf*2, x)
    # nf*(2 + 2) x 128 x 128

    dconv8 = Deconvolution(nf,
                           get_deconv_shape(batch_size, nf, 256, 256))(x)
    dconv8 = BatchNorm()(dconv8)
    x = concatenate_layers([dconv8, conv1], **merge_params)
    x = LeakyReLU(0.2)(x)
    for r in range(num_padded_conv):
        x = padded_conv(nf, x)
    # nf*(1 + 1) x 256 x 256

    dconv9 = Deconvolution(out_ch,
                           get_deconv_shape(batch_size, out_ch, 512, 512))(x)
    # out_ch x 512 x 512

    act = 'sigmoid' if is_grayscale else 'tanh'
    out = Activation(act)(dconv9)

    unet = Model(i, out, name=name)

    return unet
