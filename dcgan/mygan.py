import matplotlib as mpl
# This line allows mpl to run with no DISPLAY defined
mpl.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import os
from keras import backend as K
from keras.layers import Reshape, Flatten, LeakyReLU, Activation, Conv2D, Dense, Input
from keras.layers.convolutional import UpSampling2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate, Add
from keras.layers.core import Dropout
from keras.models import Sequential, Model
from keras.optimizers import Adam, RMSprop
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras_adversarial.image_grid_callback import ImageGridCallback
import h5py
from keras.preprocessing.image import ImageDataGenerator
import sys
sys.setrecursionlimit(10000)
sys.path.append("..")
from util.data import iterate_hdf5, Hdf5DcganIterator

from keras_adversarial import AdversarialModel, simple_gan, gan_targets
from keras_adversarial import AdversarialOptimizerSimultaneous, normal_latent_sampling
from keras_adversarial.adversarial_utils import uniform_latent_sampling
from keras_adversarial.legacy import fit, fit_generator, l1l2, AveragePooling2D
import keras.backend as K
from image_utils import dim_ordering_fix, dim_ordering_unfix, dim_ordering_shape
from skimage.io import imsave

def _model_generator():
    model = Sequential()
    nch = 256
    reg = lambda: l1l2(l1=1e-7, l2=1e-7)
    h = 5
    model.add(Dense(nch * 4 * 4, input_dim=100, W_regularizer=reg()))
    model.add(BatchNormalization(mode=0))
    model.add(Reshape(dim_ordering_shape((nch, 4, 4))))
    model.add(Convolution2D(nch / 2, h, h, border_mode='same', W_regularizer=reg()))
    model.add(BatchNormalization(mode=0, axis=1))
    model.add(LeakyReLU(0.2))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(nch / 2, h, h, border_mode='same', W_regularizer=reg()))
    model.add(BatchNormalization(mode=0, axis=1))
    model.add(LeakyReLU(0.2))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(nch / 4, h, h, border_mode='same', W_regularizer=reg()))
    model.add(BatchNormalization(mode=0, axis=1))
    model.add(LeakyReLU(0.2))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(3, h, h, border_mode='same', W_regularizer=reg()))
    model.add(Activation('sigmoid'))
    return model


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

def model_generator_with_skip(latent_dim, nch=512, h=5, initial_size=4, final_size=512, div=[2,2,4,4,8,8,16], reg=lambda: l1l2(l1=1e-7, l2=1e-7), dropout=0.):
    """
    Short skip DCGAN.
    """
    assert initial_size * (2**len(div)) == final_size
    inputs = Input(shape=(latent_dim,))
    layer = Dense(nch * initial_size * initial_size, input_dim=latent_dim, kernel_regularizer=reg())(inputs)
    layer = BatchNormalization(axis=1)(layer)
    layer = Reshape(dim_ordering_shape((nch, initial_size, initial_size)))(layer)
    div = [nch/elem for elem in div]
    for n in div:
        #              ______________________________________
        #              |                                    v
        # [layer] -> [conv] -> [bn] -> [relu] -> [conv] -> (+) -> [bn] -> [relu]
        #
        conv = Conv2D(n, h, padding='same', kernel_regularizer=reg())(layer) # conv
        bn = BatchNormalization(axis=1)(conv) # batch norm
        relu = LeakyReLU(0.2)(bn) # relu
        #conv2 = UpSampling2D(size=(2,2))(relu) # todo: could make a strided conv here
        conv2 = UpSampling2D(size=(2,2))(Conv2D(n, h, padding='same', kernel_regularizer=reg())(relu))
        conv_upsample = UpSampling2D(size=(2,2))(conv)
        merge = Add()([conv2, conv_upsample]) # concatenate
        #merge = Concatenate(axis=1)([conv2, layer_upsample]) # concatenate
        bn2 = BatchNormalization(axis=1)(merge) # batch norm 2
        relu2 = LeakyReLU(0.2)(bn2) # relu 2
        if dropout > 0:
            relu2 = Dropout(0.5)(relu2)
        layer = relu2
    layer = Conv2D(1, h, padding='same', kernel_regularizer=reg())(layer)
    layer = Activation('sigmoid')(layer)
    return Model(inputs, layer)


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


def plot_models(generator, discriminator, path):
    from keras.utils import plot_model
    if not os.path.exists(path):
        os.makedirs(path)
    for model, name in zip([generator, discriminator], ["gen", "disc"]):
        plot_model(model, show_shapes=True, to_file="%s/graph_%s.png" % (path, name))

def dump_generated_output(generator, zsamples, out_dir, prefix, batch_size=5, dump_h5=False):
    ctr = 0
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if dump_h5:
        f = h5py.File("%s/dump.h5" % out_dir, "a")
        f.create_dataset('xt', shape=(zsamples.shape[0], 512, 512, 1), dtype="float32")
        f.create_dataset('yt', shape=(zsamples.shape[0], 512, 512, 3), dtype="float32") # HACKY
    else:
        f = None
    for zbatch in iterator(zsamples, bs=batch_size):
        maps = generator.predict(zbatch)
        for i in range(maps.shape[0]):
            hm = maps[i][0]
            # normalise between 0 and 1 to get the correct brightness
            # (when you plot with matplotlib, this step is automatic)
            hm = (hm - np.min(hm)) / (np.max(hm) - np.min(hm))
            imsave(fname="%s/%s.%i.png" % (out_dir, prefix, ctr), arr=hm)
            if f != None:
                #import pdb
                #pdb.set_trace()
                f['xt'][ctr,:,:,0] = hm
            ctr += 1
            print ctr
    if f != None:
        f.flush()
        f.close()

def dump_interpolated_output(generator, zsample1, zsample2, out_name, mode='row'):
    """
    generator: generator
    zsample1: latent vector of size (latent_dim,)
    zsample2: latent vector of size (latent_dim,)
    out_name: output image, which is an image grid showing the
      interpolations.
    mode: if 'row', produce a row of interpolations. If 'matrix',
      produce a matrix of interpolations.
    returns: an output image with filename `out_name`.
    """
    from keras_adversarial import image_grid
    assert mode in ['row', 'matrix']
    if mode == 'row':
        grid = np.zeros( (1,6,512,512 ), dtype=zsample1.dtype )
    else:
        grid = np.zeros( (5,5,512,512 ), dtype=zsample1.dtype )        
    ctr = 0
    if mode == 'row':
        coefs = [0.0, 0.1, 0.3, 0.6, 0.9, 1.0]
    else:
        coefs = np.linspace(0,1,25)
    if mode == 'row':
        for a in coefs:
            tmp = generator.predict( (1-a)*zsample1[np.newaxis] + a*zsample2[np.newaxis] )[0][0]
            tmp = (tmp - np.min(tmp)) / (np.max(tmp) - np.min(tmp))
            grid[0][ctr] = tmp    
            ctr += 1
    else:
        for y in range(5):
            for x in range(5):
                a = coefs[ctr]
                tmp = generator.predict( (1-a)*zsample1[np.newaxis] + a*zsample2[np.newaxis] )[0][0]
                tmp = (tmp - np.min(tmp)) / (np.max(tmp) - np.min(tmp))
                grid[y][x] = tmp
                ctr += 1
    image_grid.write_image_grid(out_name, grid, figsize=(10,10), cmap='gray')
            
def get_model(adversarial_optimizer,
          opt_g,
          opt_d,
          generator,
          discriminator,
          loss='binary_crossentropy',
          latent_sampler=normal_latent_sampling((100,))):
    
    # gan (x - > yfake, yreal), z is gaussian generated on GPU
    # can also experiment with uniform_latent_sampling
    
    generator.summary()
    discriminator.summary()
    
    gan = simple_gan(generator=generator,
                     discriminator=discriminator,
                     latent_sampling=latent_sampler)
    # build adversarial model
    model = AdversarialModel(base_model=gan,
                             player_params=[generator.trainable_weights, discriminator.trainable_weights],
                             player_names=["generator", "discriminator"])
    model.adversarial_compile(adversarial_optimizer=adversarial_optimizer,
                              player_optimizers=[opt_g, opt_d],
                              loss=loss)
    return model


def train(model, generator, iterators, nb_epoch, path, latent_dim, save_to=None, resume=None):

    if not os.path.exists(path):
        os.makedirs(path)
    if save_to != None and not os.path.exists(save_to):
        os.makedirs(save_to)
        
    # create callback to generate images
    # TODO
    zsamples = np.random.normal(size=(10 * 10, latent_dim))

    def generator_sampler():
        #import pdb
        #pdb.set_trace()
        # [:,:,:,0] since this is a b/w image

        # latent batch size
        latent_bs = 5
        xpred_tot = []
        for i in range(0, zsamples.shape[0]//latent_bs):
            zbatch = zsamples[i*latent_bs:(i+1)*latent_bs]
            xpred_batch = dim_ordering_unfix(generator.predict(zbatch)).transpose((0, 2, 3, 1))[:,:,:,0]
            xpred_tot.append(xpred_batch)
        xpred_tot = np.asarray(xpred_tot, dtype=xpred_batch.dtype)
        return xpred_tot.reshape((10, 10) + xpred_batch.shape[1:])

    generator_cb = ImageGridCallback(os.path.join(path, "epoch-{:03d}.png"), generator_sampler, cmap='gray')

    callbacks = [generator_cb]
    # create the file logger callback
    csv_filename = "%s/results.txt" % path
    if resume == None:
        # create the file
        f = open(csv_filename, "wb")
        f.close()
    else:
        print "loading weights from... %s" % resume
        model.load_weights(resume)
    csv_logger = CSVLogger(csv_filename, append=True if os.path.exists(csv_filename) else False)
    callbacks.append(csv_logger)
    # create the model checkpointing callback
    if save_to != None:
        chkpt = ModelCheckpoint(filepath="%s/weights.{epoch:03d}.h5" % save_to, save_weights_only=True, period=5)
        callbacks.append(chkpt)
    
    it_train, it_val = iterators
    model.fit_generator(it_train, steps_per_epoch=it_train.N, epochs=nb_epoch, callbacks=callbacks)

def get_iterators(batch_size):
    dataset = h5py.File("/data/lisa/data/cbeckham/textures_v2_brown500.h5","r")
    imgen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, rotation_range=360, fill_mode="reflect")
    it_train = Hdf5DcganIterator(dataset['xt'], batch_size, imgen, is_binary=True)
    it_val = Hdf5DcganIterator(dataset['xv'], batch_size, imgen, is_binary=True)
    return it_train, it_val    

def iterator(arr, bs):
    b = 0
    while True:
        if b*bs >= arr.shape[0]:
            break
        yield arr[b*bs:(b+1)*bs]
        b += 1

if __name__ == "__main__":

    def gan_heightmap_ld1000_bigd(mode):
        latent_dim = 1000
        generator = model_generator(latent_dim, num_repeats=0)
        discriminator = model_discriminator(num_repeats=0)
        if mode == "train":
            train(AdversarialOptimizerSimultaneous(), "output/gan-heightmap-ld1000-b",
                        opt_g=Adam(1e-4, decay=1e-5),
                        opt_d=Adam(1e-3, decay=1e-5),
                        nb_epoch=100, generator=generator, discriminator=discriminator,
                        latent_dim=latent_dim)

    def gan_heightmap_ld1000_b_discbn(mode):
        # this works well! so far anyway...
        latent_dim = 1000
        generator = model_generator(latent_dim, num_repeats=0)
        discriminator = model_discriminator(num_repeats=0, bn=True)
        if mode == "train":
            train(AdversarialOptimizerSimultaneous(), "output/gan-heightmap-ld1000-b-discbn",
                        opt_g=Adam(1e-4, decay=1e-5),
                        opt_d=Adam(1e-3, decay=1e-5),
                        nb_epoch=300, generator=generator, discriminator=discriminator,
                        latent_dim=latent_dim)            

            
    def gan_heightmap_ld1000_b_discbn_i1(mode):
        assert mode in ["train", "test", "test_interp"]
        # this works well! so far anyway...
        latent_dim = 1000
        generator = model_generator(latent_dim, num_repeats=0, div=[2,2,4,4,8,8,8]) # not 16 at end
        discriminator = model_discriminator(num_repeats=0, bn=True)
        name = "gan-heightmap-ld1000-b-discbn_i1_repeat"
        model = get_model(
            adversarial_optimizer=AdversarialOptimizerSimultaneous(),
            opt_g=Adam(1e-4, decay=1e-5), opt_d=Adam(1e-3, decay=1e-5),
            generator=generator, discriminator=discriminator,
            latent_sampler=normal_latent_sampling((latent_dim,)))
        if mode == "train":
            train(model=model, generator=generator,
                  iterators=get_iterators(2), nb_epoch=300,
                  save_to="models/%s" % name,
                  path="output/%s" % name,
                  latent_dim=latent_dim,
                  resume="models/gan-heightmap-ld1000-b-discbn_i1_repeat/weights.067.h5.bak")
        elif mode == "test":
            from skimage.io import imsave
            model_name = "models/gan-heightmap-ld1000-b-discbn_i1_repeat/weights.159.h5.bak2"
            zsamples = np.random.normal(size=(10 * 10, latent_dim))
            model.load_weights(model_name)
            base_dir = "images_512/%s" % name
            dump_generated_output(generator, zsamples, base_dir)
        else:
            from skimage.io import imsave
            model_name = "models/gan-heightmap-ld1000-b-discbn_i1_repeat/weights.159.h5.bak2"
            zsamples = np.random.normal(size=(2, latent_dim))
            model.load_weights(model_name)
            dump_interpolated_output(generator, zsamples[0], zsamples[1], "images_512/%s/interp3_mat.png" % name, 'matrix')
            
            
    def gan_heightmap_ld1000_b_discbn_i1d1(mode):
        assert mode in ["train", "test"]
        # this works well! so far anyway...
        latent_dim = 1000
        generator = model_generator(latent_dim, num_repeats=0, div=[2,2,4,4,8,8,8], dropout=0.1) # not 16 at end
        discriminator = model_discriminator(num_repeats=0, bn=True)
        name = "gan-heightmap-ld1000-b-discbn_i1d1"
        model = get_model(
            adversarial_optimizer=AdversarialOptimizerSimultaneous(),
            opt_g=Adam(1e-4, decay=1e-5), opt_d=Adam(1e-3, decay=1e-5),
            generator=generator, discriminator=discriminator,
            latent_sampler=normal_latent_sampling((latent_dim,)))
        if mode == "train":
            train(model=model, generator=generator,
                  iterators=get_iterators(2), nb_epoch=500,
                  save_to="models/%s" % name,
                  path="output/%s" % name,
                  latent_dim=latent_dim)
                       

    def gan_heightmap_ld1000_b_discbn_i1ls(mode):
        assert mode in ["train", "test"]
        # this works well! so far anyway...
        latent_dim = 1000
        generator = model_generator(latent_dim, num_repeats=0, div=[2,2,4,4,8,8,8]) # not 16 at end
        discriminator = model_discriminator(num_repeats=0, bn=True, nonlinearity='linear')
        name = "gan-heightmap-ld1000-b-discbn_i1ls"
        model = get_model(
            adversarial_optimizer=AdversarialOptimizerSimultaneous(),
            opt_g=Adam(1e-4, decay=1e-5), opt_d=Adam(1e-3, decay=1e-5),
            generator=generator, discriminator=discriminator,
            latent_sampler=normal_latent_sampling((latent_dim,)), loss='mean_squared_error')
        if mode == "train":
            train(model=model, generator=generator,
                  iterators=get_iterators(2), nb_epoch=300,
                  save_to="models/%s" % name,
                  path="output/%s" % name,
                  latent_dim=latent_dim)


    def gan_heightmap_ld1000_b_discbn_i1ls_rmspropd(mode):
        assert mode in ["train", "test"]
        # this works well! so far anyway...
        latent_dim = 1000
        generator = model_generator(latent_dim, num_repeats=0, div=[2,2,4,4,8,8,8]) # not 16 at end
        discriminator = model_discriminator(num_repeats=0, bn=True, nonlinearity='linear')
        name = "gan-heightmap-ld1000-b-discbn_i1ls_rmspropd"
        model = get_model(
            adversarial_optimizer=AdversarialOptimizerSimultaneous(),
            opt_g=RMSprop(1e-4, decay=1e-5), opt_d=RMSprop(1e-4, decay=1e-5),
            generator=generator, discriminator=discriminator,
            latent_sampler=normal_latent_sampling((latent_dim,)), loss='mean_squared_error')
        if mode == "train":
            train(model=model, generator=generator,
                  iterators=get_iterators(2), nb_epoch=300,
                  save_to="models/%s" % name,
                  path="output/%s" % name,
                  latent_dim=latent_dim)

    def gan_heightmap_ld1000_b_discbn_i1ls_rmspropd_weakd(mode):
        assert mode in ["train", "test"]
        # this works well! so far anyway...
        latent_dim = 1000
        generator = model_generator(latent_dim, num_repeats=0, div=[2,2,4,4,8,8,8]) # not 16 at end
        discriminator = model_discriminator(num_repeats=0, bn=True, nonlinearity='linear',div=[8,4,4,4,2,2,2])
        name = "gan-heightmap-ld1000-b-discbn_i1ls_rmspropd_weakd"
        model = get_model(
            adversarial_optimizer=AdversarialOptimizerSimultaneous(),
            opt_g=RMSprop(1e-4, decay=1e-5), opt_d=RMSprop(1e-4, decay=1e-5),
            generator=generator, discriminator=discriminator,
            latent_sampler=normal_latent_sampling((latent_dim,)), loss='mean_squared_error')
        if mode == "train":
            train(model=model, generator=generator,
                  iterators=get_iterators(2), nb_epoch=300,
                  save_to="models/%s" % name,
                  path="output/%s" % name,
                  latent_dim=latent_dim)
        else:
            model_name = "models/%s/weights.069.h5.bak" % name
            zsamples = np.random.normal(size=(10 * 10, latent_dim))
            model.load_weights(model_name)
            base_dir = "images_512/%s" % name
            dump_generated_output(generator, zsamples, base_dir, os.path.basename(model_name), dump_h5=True)


    def gan_heightmap_ld1000_b_discbn_i1ls_rmspropd_weakd2(mode):
        assert mode in ["train", "test"]
        # this works well! so far anyway...
        latent_dim = 1000
        generator = model_generator(latent_dim, num_repeats=0, div=[2,2,4,4,8,8,8]) # not 16 at end
        discriminator = model_discriminator(num_repeats=0, bn=True, nonlinearity='linear',div=[8,8,6,6,4,4,2])
        name = "gan-heightmap-ld1000-b-discbn_i1ls_rmspropd_weakd2"
        model = get_model(
            adversarial_optimizer=AdversarialOptimizerSimultaneous(),
            opt_g=RMSprop(1e-4, decay=1e-5), opt_d=RMSprop(1e-4, decay=1e-5),
            generator=generator, discriminator=discriminator,
            latent_sampler=normal_latent_sampling((latent_dim,)), loss='mean_squared_error')
        if mode == "train":
            train(model=model, generator=generator,
                  iterators=get_iterators(2), nb_epoch=300,
                  save_to="models/%s" % name,
                  path="output/%s" % name,
                  latent_dim=latent_dim)


            
    def gan_heightmap_ld1000_b_discbn_i1ls_skip_weakd(mode):
        assert mode in ["train", "test"]
        latent_dim = 1000
        generator = model_generator_with_skip(latent_dim, div=[2,2,4,4,8,8,8]) # not 16 at end
        discriminator = model_discriminator(num_repeats=0, bn=True, nonlinearity='linear', div=[8,4,4,4,2,2,2])
        name = "gan-heightmap-ld1000-b-discbn_i1_skip_weakd"
        model = get_model(
            adversarial_optimizer=AdversarialOptimizerSimultaneous(),
            opt_g=Adam(1e-4, decay=1e-5), opt_d=Adam(1e-3, decay=1e-5),
            generator=generator, discriminator=discriminator,
            latent_sampler=normal_latent_sampling((latent_dim,)),loss='mean_squared_error')
        plot_models(generator, discriminator, "output/%s" % name)
        if mode == "train":
            train(model=model, generator=generator,
                  iterators=get_iterators(2), nb_epoch=300,
                  save_to="models/%s" % name,
                  path="output/%s" % name,
                  latent_dim=latent_dim)

    def gan_heightmap_ld1000_b_discbn_i1_skip2(mode):
        assert mode in ["train", "test"]
        latent_dim = 1000
        generator = model_generator_with_skip(latent_dim, div=[2,4,4,8,8,16,16]) # not 16 at end
        discriminator = model_discriminator(num_repeats=0, bn=True)
        name = "gan-heightmap-ld1000-b-discbn_i1_skip2"
        model = get_model(
            adversarial_optimizer=AdversarialOptimizerSimultaneous(),
            opt_g=Adam(1e-4, decay=1e-5), opt_d=Adam(1e-3, decay=1e-5),
            generator=generator, discriminator=discriminator,
            latent_sampler=normal_latent_sampling((latent_dim,)))
        plot_models(generator, discriminator, "output/%s" % name)
        if mode == "train":
            train(model=model, generator=generator,
                  iterators=get_iterators(2), nb_epoch=300,
                  save_to="models/%s" % name,
                  path="output/%s" % name,
                  latent_dim=latent_dim)

    def gan_heightmap_ld1000_b_discbn_i1_skip3(mode):
        assert mode in ["train", "test"]
        # intended to be somewhat way higher capacity than
        # the corresponding discriminator...
        latent_dim = 1000
        generator = model_generator_with_skip(latent_dim, div=[2,2,4,4,8,8,16]) # not 16 at end
        discriminator = model_discriminator(num_repeats=0, bn=True)
        name = "gan-heightmap-ld1000-b-discbn_i1_skip3"
        model = get_model(
            adversarial_optimizer=AdversarialOptimizerSimultaneous(),
            opt_g=Adam(1e-4, decay=1e-5), opt_d=Adam(1e-3, decay=1e-5),
            generator=generator, discriminator=discriminator,
            latent_sampler=normal_latent_sampling((latent_dim,)))
        plot_models(generator, discriminator, "output/%s" % name)
        if mode == "train":
            train(model=model, generator=generator,
                  iterators=get_iterators(2), nb_epoch=300,
                  save_to="models/%s" % name,
                  path="output/%s" % name,
                  latent_dim=latent_dim)

    def gan_heightmap_ld1000_b_discbn_i1_skip4(mode):
        assert mode in ["train", "test"]
        # intended to be somewhat way higher capacity than
        # the corresponding discriminator...
        latent_dim = 1000
        generator = model_generator_with_skip(latent_dim, div=[1,2,2,2,4,4,4]) # not 16 at end
        discriminator = model_discriminator(num_repeats=0, bn=True)
        name = "gan-heightmap-ld1000-b-discbn_i1_skip4"
        model = get_model(
            adversarial_optimizer=AdversarialOptimizerSimultaneous(),
            opt_g=Adam(1e-4, decay=1e-5), opt_d=Adam(1e-3, decay=1e-5),
            generator=generator, discriminator=discriminator,
            latent_sampler=normal_latent_sampling((latent_dim,)))
        plot_models(generator, discriminator, "output/%s" % name)
        if mode == "train":
            train(model=model, generator=generator,
                  iterators=get_iterators(2), nb_epoch=300,
                  save_to="models/%s" % name,
                  path="output/%s" % name,
                  latent_dim=latent_dim)

    def gan_heightmap_ld1000_b_discbn_i1_skip4ls_rmsprop(mode):
        assert mode in ["train", "test"]
        # intended to be somewhat way higher capacity than
        # the corresponding discriminator...
        latent_dim = 1000
        generator = model_generator_with_skip(latent_dim, div=[1,2,2,2,4,4,4]) # not 16 at end
        discriminator = model_discriminator(num_repeats=0, bn=True)
        name = "gan-heightmap-ld1000-b-discbn_i1_skip4ls_rmsprop"
        model = get_model(
            adversarial_optimizer=AdversarialOptimizerSimultaneous(),
            opt_g=RMSprop(1e-3), opt_d=RMSprop(1e-3),
            generator=generator, discriminator=discriminator,
            latent_sampler=normal_latent_sampling((latent_dim,)))
        plot_models(generator, discriminator, "output/%s" % name)
        if mode == "train":
            train(model=model, generator=generator,
                  iterators=get_iterators(2), nb_epoch=300,
                  save_to="models/%s" % name,
                  path="output/%s" % name,
                  latent_dim=latent_dim)


            
    def gan_heightmap_ld1000_b_discbn_i1_skip5(mode):
        assert mode in ["train", "test"]
        # intended to be somewhat way higher capacity than
        # the corresponding discriminator...
        latent_dim = 1000
        generator = model_generator_with_skip(latent_dim, initial_size=2, div=[1,2,2,2,4,4,4,4]) # not 16 at end
        discriminator = model_discriminator(num_repeats=0, bn=True)
        name = "gan-heightmap-ld1000-b-discbn_i1_skip5"
        model = get_model(
            adversarial_optimizer=AdversarialOptimizerSimultaneous(),
            opt_g=Adam(1e-4, decay=1e-5), opt_d=Adam(1e-3, decay=1e-5),
            generator=generator, discriminator=discriminator,
            latent_sampler=normal_latent_sampling((latent_dim,)))
        plot_models(generator, discriminator, "output/%s" % name)
        if mode == "train":
            train(model=model, generator=generator,
                  iterators=get_iterators(2), nb_epoch=300,
                  save_to="models/%s" % name,
                  path="output/%s" % name,
                  latent_dim=latent_dim)

    def gan_heightmap_ld1000_b_discbn_i1_skip6(mode):
        assert mode in ["train", "test"]
        # intended to be somewhat way higher capacity than
        # the corresponding discriminator...
        latent_dim = 1000
        generator = model_generator_with_skip(latent_dim, nch=1024, initial_size=2, div=[2,4,4,8,8,16,16,32]) # not 16 at end
        discriminator = model_discriminator(num_repeats=0, bn=True)
        name = "gan-heightmap-ld1000-b-discbn_i1_skip6"
        model = get_model(
            adversarial_optimizer=AdversarialOptimizerSimultaneous(),
            opt_g=Adam(1e-4, decay=1e-5), opt_d=Adam(1e-3, decay=1e-5),
            generator=generator, discriminator=discriminator,
            latent_sampler=normal_latent_sampling((latent_dim,)))
        plot_models(generator, discriminator, "output/%s" % name)
        if mode == "train":
            train(model=model, generator=generator,
                  iterators=get_iterators(2), nb_epoch=300,
                  save_to="models/%s" % name,
                  path="output/%s" % name,
                  latent_dim=latent_dim)
            
    def gan_heightmap_ld1000_b_discbn_i1_skip6eq(mode):
        assert mode in ["train", "test"]
        # intended to be somewhat way higher capacity than
        # the corresponding discriminator...
        latent_dim = 1000
        generator = model_generator_with_skip(latent_dim, nch=1024, initial_size=2, div=[2,4,4,8,8,16,16,32]) # not 16 at end
        discriminator = model_discriminator(num_repeats=0, bn=True)
        name = "gan-heightmap-ld1000-b-discbn_i1_skip6eq"
        model = get_model(
            adversarial_optimizer=AdversarialOptimizerSimultaneous(),
            opt_g=Adam(1e-3, decay=1e-5), opt_d=Adam(1e-3, decay=1e-5),
            generator=generator, discriminator=discriminator,
            latent_sampler=normal_latent_sampling((latent_dim,)))
        plot_models(generator, discriminator, "output/%s" % name)
        if mode == "train":
            train(model=model, generator=generator,
                  iterators=get_iterators(2), nb_epoch=300,
                  save_to="models/%s" % name,
                  path="output/%s" % name,
                  latent_dim=latent_dim)


    def gan_heightmap_ld1000_b_discbn_i1_skip6ls(mode):
        assert mode in ["train", "test"]
        # intended to be somewhat way higher capacity than
        # the corresponding discriminator...
        latent_dim = 1000
        generator = model_generator_with_skip(latent_dim, nch=1024, initial_size=2, div=[2,4,4,8,8,16,16,32]) # not 16 at end
        discriminator = model_discriminator(num_repeats=0, bn=True, nonlinearity='linear')
        name = "gan-heightmap-ld1000-b-discbn_i1_skip6ls"
        model = get_model(
            adversarial_optimizer=AdversarialOptimizerSimultaneous(),
            opt_g=Adam(1e-4, decay=1e-5), opt_d=Adam(1e-3, decay=1e-5),
            generator=generator, discriminator=discriminator,
            latent_sampler=normal_latent_sampling((latent_dim,)), loss='mean_squared_error')
        plot_models(generator, discriminator, "output/%s" % name)
        if mode == "train":
            train(model=model, generator=generator,
                  iterators=get_iterators(2), nb_epoch=300,
                  save_to="models/%s" % name,
                  path="output/%s" % name,
                  latent_dim=latent_dim)

    def gan_heightmap_ld1000_b_discbn_i1_skip7ls_rmsprop(mode):
        assert mode in ["train", "test"]
        # intended to be somewhat way higher capacity than
        # the corresponding discriminator...
        latent_dim = 1000
        generator = model_generator_with_skip(latent_dim, nch=1024, initial_size=4, div=[2,4,4,8,8,16,16]) # not 16 at end
        discriminator = model_discriminator(num_repeats=0, bn=True, nonlinearity='linear')
        name = "gan-heightmap-ld1000-b-discbn_i1_skip7ls_rmsprop"
        # f0k's lsgan uses rmsprop with 1e-4, with no LR decay
        model = get_model(
            adversarial_optimizer=AdversarialOptimizerSimultaneous(),
            opt_g=RMSprop(1e-4), opt_d=RMSprop(1e-4),
            generator=generator, discriminator=discriminator,
            latent_sampler=normal_latent_sampling((latent_dim,)), loss='mean_squared_error')
        plot_models(generator, discriminator, "output/%s" % name)
        if mode == "train":
            train(model=model, generator=generator,
                  iterators=get_iterators(2), nb_epoch=300,
                  save_to="models/%s" % name,
                  path="output/%s" % name,
                  latent_dim=latent_dim)


    def gan_heightmap_ld1000_b_discbn_i1_skip7lsd5_rmsprop(mode):
        assert mode in ["train", "test"]
        # intended to be somewhat way higher capacity than
        # the corresponding discriminator...
        latent_dim = 1000
        generator = model_generator_with_skip(latent_dim, nch=1024, initial_size=4, div=[2,4,4,8,8,16,16], dropout=0.5) # not 16 at end
        discriminator = model_discriminator(num_repeats=0, bn=True, nonlinearity='linear')
        name = "gan-heightmap-ld1000-b-discbn_i1_skip7lsd5_rmsprop"
        # f0k's lsgan uses rmsprop with 1e-4, with no LR decay
        model = get_model(
            adversarial_optimizer=AdversarialOptimizerSimultaneous(),
            opt_g=RMSprop(1e-4), opt_d=RMSprop(1e-4),
            generator=generator, discriminator=discriminator,
            latent_sampler=normal_latent_sampling((latent_dim,)), loss='mean_squared_error')
        plot_models(generator, discriminator, "output/%s" % name)
        if mode == "train":
            train(model=model, generator=generator,
                  iterators=get_iterators(2), nb_epoch=300,
                  save_to="models/%s" % name,
                  path="output/%s" % name,
                  latent_dim=latent_dim)

            


            
    def gan_heightmap_ld1000_b_discbn_i1_skip6ls_ld2000(mode):
        assert mode in ["train", "test"]
        # intended to be somewhat way higher capacity than
        # the corresponding discriminator...
        latent_dim = 2000
        generator = model_generator_with_skip(latent_dim, nch=1024, initial_size=2, div=[2,4,4,8,8,16,16,32]) # not 16 at end
        discriminator = model_discriminator(num_repeats=0, bn=True, nonlinearity='linear')
        name = "gan-heightmap-ld1000-b-discbn_i1_skip6ls_ld2000"
        model = get_model(
            adversarial_optimizer=AdversarialOptimizerSimultaneous(),
            opt_g=Adam(1e-4, decay=1e-5), opt_d=Adam(1e-3, decay=1e-5),
            generator=generator, discriminator=discriminator,
            latent_sampler=normal_latent_sampling((latent_dim,)), loss='mean_squared_error')
        plot_models(generator, discriminator, "output/%s" % name)
        if mode == "train":
            train(model=model, generator=generator,
                  iterators=get_iterators(2), nb_epoch=300,
                  save_to="models/%s" % name,
                  path="output/%s" % name,
                  latent_dim=latent_dim)


            
    def gan_heightmap_ld1000_b_discbn_i1_skip6lseq(mode):
        assert mode in ["train", "test"]
        # intended to be somewhat way higher capacity than
        # the corresponding discriminator...
        latent_dim = 1000
        generator = model_generator_with_skip(latent_dim, nch=1024, initial_size=2, div=[2,4,4,8,8,16,16,32]) # not 16 at end
        discriminator = model_discriminator(num_repeats=0, bn=True, nonlinearity='linear')
        name = "gan-heightmap-ld1000-b-discbn_i1_skip6lseq"
        model = get_model(
            adversarial_optimizer=AdversarialOptimizerSimultaneous(),
            opt_g=Adam(1e-3, decay=1e-5), opt_d=Adam(1e-3, decay=1e-5),
            generator=generator, discriminator=discriminator,
            latent_sampler=normal_latent_sampling((latent_dim,)), loss='mean_squared_error')
        plot_models(generator, discriminator, "output/%s" % name)
        if mode == "train":
            train(model=model, generator=generator,
                  iterators=get_iterators(2), nb_epoch=300,
                  save_to="models/%s" % name,
                  path="output/%s" % name,
                  latent_dim=latent_dim)

            
                    
    def gan_heightmap_ld1000_b_discbn_bb1(mode):
        # may do well...
        latent_dim = 1000
        generator = model_generator(latent_dim, num_repeats=1)
        discriminator = model_discriminator(num_repeats=1, bn=True, div=[4,4,4,2,2,2,1])
        if mode == "train":
            train(AdversarialOptimizerSimultaneous(), "output/gan-heightmap-ld1000-b-discbn-bb1",
                        opt_g=Adam(1e-4, decay=1e-5),
                        opt_d=Adam(1e-3, decay=1e-5),
                        nb_epoch=300, generator=generator, discriminator=discriminator,
                        latent_dim=latent_dim, iterators=get_iterators(2))

    def gan_heightmap_ld1000_b_discbn_bb2(mode):
        # disc has slightly higher capacity than bb1
        latent_dim = 1000
        generator = model_generator(latent_dim, num_repeats=1)
        discriminator = model_discriminator(num_repeats=1, bn=True, div=[4,4,4,4,2,2,2,1])
        if mode == "train":
            train(AdversarialOptimizerSimultaneous(), "output/gan-heightmap-ld1000-b-discbn-bb2",
                        opt_g=Adam(1e-4, decay=1e-5),
                        opt_d=Adam(1e-3, decay=1e-5),
                        nb_epoch=300, generator=generator, discriminator=discriminator,
                        latent_dim=latent_dim, iterators=get_iterators(2))

    def gan_heightmap_ld1000_b_discbn_bb2b(mode):
        # disc has slightly higher capacity than bb1
        # but also equal LRs
        latent_dim = 1000
        generator = model_generator(latent_dim, num_repeats=1)
        discriminator = model_discriminator(num_repeats=1, bn=True, div=[4,4,4,4,2,2,2,1])
        if mode == "train":
            train(AdversarialOptimizerSimultaneous(), "output/gan-heightmap-ld1000-b-discbn-bb2b",
                        opt_g=Adam(1e-3, decay=1e-5),
                        opt_d=Adam(1e-3, decay=1e-5),
                        nb_epoch=300, generator=generator, discriminator=discriminator,
                        latent_dim=latent_dim, iterators=get_iterators(3))


            

            
    def gan_heightmap_ld1000_b_discbn_unif(mode):
        # try out unif sampling
        # https://arxiv.org/pdf/1706.00082.pdf
        # ^ those guys also limit the range of the uniform to control for
        # artifacts... may be useful
        latent_dim = 1000
        generator = model_generator(latent_dim, num_repeats=0)
        discriminator = model_discriminator(num_repeats=0, bn=True)
        if mode == "train":
            train(AdversarialOptimizerSimultaneous(), "output/gan-heightmap-ld1000-b-discbn-unif",
                        opt_g=Adam(1e-4, decay=1e-5),
                        opt_d=Adam(1e-3, decay=1e-5),
                        nb_epoch=100, generator=generator, discriminator=discriminator,
                        latent_dim=latent_dim,
                        latent_sampler=uniform_latent_sampling((latent_dim,)))


            
    locals()[ sys.argv[1] ]( sys.argv[2] )
