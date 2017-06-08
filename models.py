__doc__ = """The model definitions for the pix2pix network taken from the
retina repository at https://github.com/costapt/vess2ret
"""
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

## TODO: clean this mess up

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


    
def discriminator(a_ch, b_ch, nf, extra_depth=0, opt=Adam(lr=2e-4, beta_1=0.5), name='d'):
    """Define the discriminator network.

    Parameters:
    - a_ch: the number of channels of the first image;
    - b_ch: the number of channels of the second image;
    - nf: the number of filters of the first layer.
    - extra_depth: by default, the discriminator's output is 16x16,
      but we can manually add more depth to make the patches bigger,
      e.g. `extra_depth=1` means 8x8, `extra_depth=2` means 4x4, etc.
    """
    i = Input(shape=(a_ch + b_ch, 512, 512))

    # (a_ch + b_ch) x 512 x 512
    # nf x 256 x 256
    # nf*2 x 128 x 128
    # nf*4 x 64 x 64    
    # nf*8 x 32 x 32

    x = i
    default_depths = [1,2,4,8]
    for depth_factor in default_depths:
        x = Convolution(nf*depth_factor)(x)
        x = LeakyReLU(0.2)(x)

    for r in range(extra_depth):
        x = Convolution( nf*(2**(3+r+1)) )(x)
        x = LeakyReLU(0.2)(x)

    x = Convolution(1)(x)
    out = Activation('sigmoid')(x)
    # 1 x 16 x 16

    d = Model(i, out, name=name)

    def d_loss(y_true, y_pred):
        L = objectives.binary_crossentropy(K.batch_flatten(y_true),
                                           K.batch_flatten(y_pred))
        return L

    d.compile(optimizer=opt, loss=d_loss)
    return d


def pix2pix(atob, d, a_ch, b_ch, alpha=100, is_a_binary=False,
            is_b_binary=False, opt=Adam(lr=2e-4, beta_1=0.5), reconstruction_only=False, name='pix2pix'):
    # type: (...) -> keras.models.Model
    """
    Define the pix2pix network.
    :param atob:
    :param d:
    :param a_ch:
    :param b_ch:
    :param alpha:
    :param is_a_binary:
    :param is_b_binary:
    :param opt:
    :param name:
    :return:
    """
    a = Input(shape=(a_ch, 512, 512))
    b = Input(shape=(b_ch, 512, 512))

    # A -> B'
    bp = atob(a)

    # Discriminator receives the pair of images
    d_in = concatenate_layers([a, bp], mode='concat', concat_axis=1)

    pix2pix = Model([a, b], d(d_in), name=name)

    def pix2pix_loss(y_true, y_pred):
        y_true_flat = K.batch_flatten(y_true)
        y_pred_flat = K.batch_flatten(y_pred)

        # Adversarial Loss
        L_adv = objectives.binary_crossentropy(y_true_flat, y_pred_flat)

        # A to B loss
        b_flat = K.batch_flatten(b)
        bp_flat = K.batch_flatten(bp)
        if is_b_binary:
            L_atob = objectives.binary_crossentropy(b_flat, bp_flat)
        else:
            L_atob = K.mean(K.abs(b_flat - bp_flat))

        if reconstruction_only:
            return L_atob
        else:
            return L_adv + alpha * L_atob

    # This network is used to train the generator. Freeze the discriminator part.
    pix2pix.get_layer('d').trainable = False

    pix2pix.compile(optimizer=opt, loss=pix2pix_loss)
    return pix2pix


if __name__ == '__main__':
    import doctest

    TEST_TF = True
    if TEST_TF:
        os.environ['KERAS_BACKEND'] = 'tensorflow'
    else:
        os.environ['KERAS_BACKEND'] = 'theano'
    doctest.testsource('models.py', verbose=True, optionflags=doctest.ELLIPSIS)
