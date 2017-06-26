"""The script used to train the model."""
import os
import sys
import getopt

import numpy as np
import models as m

from tqdm import tqdm
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from util.data import TwoImageIterator, iterate_hdf5, Hdf5Iterator
from util.util import MyDict, log, save_weights, load_weights, load_losses, create_expt_dir, convert_to_rgb

import h5py
from time import time

from collections import OrderedDict

import sys
sys.setrecursionlimit(10000)

def discriminator_generator(it, atob, dout_size):
    """
    Generate batches for the discriminator.

    Parameters:
    - it: an iterator that returns a pair of images;
    - atob: the generator network that maps an image to another representation;
    - dout_size: the size of the output of the discriminator.
    """
    while True:
        a_real, b_real = it.next()
        a_fake = np.copy(a_real)
        b_fake = atob.predict(a_fake)
        #b_fake = np.zeros_like(b_real)

        # Concatenate the channels. Images become (ch_a + ch_b) x 256 x 256
        fake = np.concatenate((a_fake, b_fake), axis=1)
        real = np.concatenate((a_real, b_real), axis=1)

        # Concatenate fake and real pairs into a single batch
        batch_x = np.concatenate((fake, real), axis=0)

        # 1 is fake, 0 is real
        batch_y = np.ones((batch_x.shape[0], 1) + dout_size)
        batch_y[fake.shape[0]:] = 0
        
        yield batch_x, batch_y


        
def train_discriminator(d, it, steps_per_epoch):
    """Train the discriminator network."""
    #return d.fit_generator(it, steps_per_epoch=steps_per_epoch, epochs=1, verbose=False, workers=1)
    losses = []
    for b in range(steps_per_epoch):
        x, y = next(it)
        losses.append(d.train_on_batch(x,y))
    return np.mean(losses)

def pix2pix_generator(it, dout_size):
    """
    Generate data for the generator network.

    Parameters:
    - it: an iterator that returns a pair of images;
    - dout_size: the size of the output of the discriminator.
    """
    while True:
        for a, b in it:
            # 1 is fake, 0 is real
            y = np.zeros((a.shape[0], 1) + dout_size)
            #import pdb
            #pdb.set_trace()
            yield [a, b], y


def train_pix2pix(pix2pix, it, steps_per_epoch):
    """Train the generator network."""
    #return pix2pix.fit_generator(it, steps_per_epoch=steps_per_epoch, epochs=1, verbose=False, workers=1)
    losses = []
    for b in range(steps_per_epoch):
        x, y = next(it)
        losses.append(pix2pix.train_on_batch(x,y))
    return np.mean(losses)

def evaluate_generator(fn, it, steps_per_epoch):
    """
    fn: model to evaluate
    it: iterator to get x,y from
    steps_per_epoch: steps of the iterator to run
    """
    losses = []
    for b in range(steps_per_epoch):
        x, y = next(it)
        losses.append( fn.test_on_batch(x, y) )
    return np.mean(losses)


def evaluate(models, generators, losses, params):
    """Evaluate and display the losses of the models."""
    # Get necessary generators
    d_gen = generators.d_gen_val
    p2p_gen = generators.p2p_gen_val

    # Get necessary models
    d = models.d
    p2p = models.p2p

    # Evaluate
    d_loss = evaluate_generator(d, d_gen, steps_per_epoch=params.val_samples // params.batch_size)
    p2p_loss = evaluate_generator(p2p, p2p_gen, steps_per_epoch=params.val_samples // params.batch_size)

    losses['d_val'].append(d_loss)
    losses['p2p_val'].append(p2p_loss)

    print ''
    print ('Train Losses of (D={0} / P2P={1});\n'
           'Validation Losses of (D={2} / P2P={3})'.format(
                losses['d'][-1], losses['p2p'][-1], d_loss, p2p_loss))

    return d_loss, p2p_loss


def model_creation(d, atob, params):
    """Create all the necessary models."""
    opt = Adam(lr=params.lr, beta_1=params.beta_1)
    p2p = m.pix2pix(atob, d, params.a_ch, params.b_ch, alpha=params.alpha, opt=opt,
                    is_a_grayscale=params.is_a_grayscale, is_b_grayscale=params.is_b_grayscale, reconstruction_only=params.reconstruction_only)

    models = MyDict({
        'atob': atob,
        'd': d,
        'p2p': p2p,
    })

    for key in models:
        models[key].summary()
        #models[key]._make_predict_function() ## ???
    
    return models


def generators_creation(it_train, it_val, models, dout_size):
    """Create all the necessary data generators."""
    # Discriminator data generators
    d_gen = discriminator_generator(it_train, models.atob, dout_size)
    d_gen_val = discriminator_generator(it_val, models.atob, dout_size)
    
    # Workaround to make tensorflow work. When atob.predict is called the first
    # time it calls tf.get_default_graph. This should be done on the main thread
    # and not inside fit_generator. See https://github.com/fchollet/keras/issues/2397
    #next(d_gen)

    # pix2pix data generators
    p2p_gen = pix2pix_generator(it_train, dout_size)
    p2p_gen_val = pix2pix_generator(it_val, dout_size)

    generators = MyDict({
        'd_gen': d_gen,
        'd_gen_val': d_gen_val,
        'p2p_gen': p2p_gen,
        'p2p_gen_val': p2p_gen_val,
    })

    return generators


def train_iteration(models, generators, losses, params):
    """Perform a train iteration."""
    # Get necessary generators
    d_gen = generators.d_gen
    p2p_gen = generators.p2p_gen

    # Get necessary models
    d = models.d
    p2p = models.p2p

    # Update the dscriminator
    if not params.reconstruction_only:
        dhist = train_discriminator(d, d_gen, steps_per_epoch=params.train_samples // params.batch_size)
        print "  dhist = ", dhist
        #losses['d'].extend(dhist.history['loss'])
    else:
        dhist = -1
    losses['d'].append(dhist)

    # Update the generator
    p2phist = train_pix2pix(p2p, p2p_gen, steps_per_epoch=params.val_samples // params.batch_size)
    print "  p2phist = ", p2phist
    #losses['p2p'].extend(p2phist.history['loss'])
    losses['p2p'].append(p2phist)
    

def train(models, it_train, it_val, params):
    """
    Train the model.

    Parameters:
    - models: a dictionary with all the models.
        - atob: a model that goes from A to B.
        - d: the discriminator model.
        - p2p: a Pix2Pix model.
    - it_train: the iterator of the training data.
    - it_val: the iterator of the validation data.
    - params: parameters of the training procedure.
    - dout_size: the size of the output of the discriminator model.
    """
    # Create the experiment folder and save the parameters
    create_expt_dir(params)

    # save model graphs
    from keras.utils import plot_model
    for key in models:
        plot_model(models[key], show_shapes=True, to_file="%s/%s/graph_%s.png" % (params.log_dir, params.expt_name, key))
    
    # Get the output shape of the discriminator
    dout_size = models.d.output_shape[-2:]
    # Define the data generators
    generators = generators_creation(it_train, it_val, models, dout_size)

    # Define the number of samples to use on each training epoch
    params.train_samples = it_train.N
    train_batches_per_epoch = params.train_samples // params.batch_size

    # Define the number of samples to use for validation
    params.val_samples = it_val.N
    val_batches_per_epoch = params.val_samples // params.batch_size

    keys = ["epoch", "p2p", "d", "p2p_val", "d_val","time"]
    losses = OrderedDict({})
    for key in keys: losses[key] = []
    results_filename = "%s/%s/results.txt" % (params.log_dir, params.expt_name)
    results_flag = "a" if params.continue_train else "wb"
    file_handles = {
        'results': open(results_filename, results_flag)
    }
    if results_flag == "wb":
        file_handles['results'].write( ",".join(losses.keys()) + "\n" )
        file_handles['results'].flush()
    
    #if params.continue_train:
    #    losses = load_losses(log_dir=params.log_dir, expt_name=params.expt_name)
        
    for e in range(params.epochs):
        t0 = time()
        losses["epoch"].append(e+1)

        #for b in range(train_batches_per_epoch):
        train_iteration(models, generators, losses, params)

        # Evaluate how the models is doing on the validation set.
        evaluate(models, generators, losses, params)

        losses["time"].append(time()-t0)
        
        if (e + 1) % params.save_every == 0:
            save_weights(models, log_dir=params.log_dir, expt_name=params.expt_name)
            log(file_handles, losses, models.atob, it_val, log_dir=params.log_dir, expt_name=params.expt_name,
                is_a_grayscale=params.is_a_grayscale, is_b_grayscale=params.is_b_grayscale)

if __name__ == '__main__':
    a = sys.argv[1:]

    params = MyDict({
        # Model
        'alpha': 100,  # The weight of the reconstruction loss of the atob model
        # Train
        'epochs': 100,  # Number of epochs to train the model
        'batch_size': 1,  # The batch size
        'save_every': 1,  # Save results every 'save_every' epochs on the log folder
        'lr': 2e-4,  # The learning rate to train the models
        'beta_1': 0.5,  # The beta_1 value of the Adam optimizer
        'continue_train': False,  # If it should continue the training from the last checkpoint
        # File system
        'log_dir': 'log',  # Directory to log
        'expt_name': None,  # The name of the experiment. Saves the logs into a folder with this name
        'load_to_memory': True,  # Whether to load the images into memory
        # Image
        'dataset': "/data/lisa/data/cbeckham/textures_v2_90-10.h5", # HDF5 file for training/test data
        'a_ch': 1,  # Number of channels of images A
        'b_ch': 3,  # Number of channels of images B
        'is_a_grayscale': True,  # If A is grayscale, the image will only have one channel
        'is_b_grayscale': False,  # If B is grayscale, the image will only have one channel
        'target_size': 512,  # The size of the images loaded by the iterator. DOES NOT CHANGE THE MODELS
        'reconstruction_only':False
    })

    def print_params():
        print "params:"
        for key in params:
            print key, ":", params[key]
    

    def get_iterators(da=True):
        dataset = h5py.File(params.dataset,"r")
        if da:
            imgen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, rotation_range=360, fill_mode="reflect")
        else:
            imgen = ImageDataGenerator()
        it_train = Hdf5Iterator(dataset['xt'], dataset['yt'], params.batch_size, imgen, is_a_grayscale=params.is_a_grayscale, is_b_grayscale=params.is_b_grayscale)
        it_val = Hdf5Iterator(dataset['xv'], dataset['yv'], params.batch_size, imgen, is_a_grayscale=params.is_a_grayscale, is_b_grayscale=params.is_b_grayscale)
        return it_train, it_val


    def idk4(mode,seed):
        np.random.seed(seed)
        # override params here
        params.expt_name = "idk4_2_alpha100d"
        params.dataset = "/data/lisa/data/cbeckham/textures_v2_brown500.h5"
        params.batch_size = 8
        params.epochs = 1000
        params.alpha = 100.
        params.reconstruction_only = False
        params.lr = 1e-3
        params.continue_train = True
        print_params()
        dopt = Adam(lr=params.lr, beta_1=params.beta_1)
        from architectures import default_models
        # Define the generator
        unet = default_models.g_unet(in_ch=params.a_ch,
                             out_ch=params.b_ch,
                             nf=64,
                             num_padded_conv=2,
                             batch_size=params.batch_size,
                             is_grayscale=params.is_b_grayscale)
        # Define the discriminator
        d = m.discriminator(params.a_ch, params.b_ch, 1, depths=[1,2,4], bn=True, opt=dopt) #[1,2,4,8,16,24,32,40] 
        if params.continue_train:
            print "loading weights..."
            load_weights(unet, d, log_dir=params.log_dir, expt_name=params.expt_name)
        models = model_creation(d, unet, params)
        it_train, it_val = get_iterators()
        if mode == "train":
            train(models, it_train, it_val, params)


    def idk4e(mode,seed):
        np.random.seed(seed)
        # override params here
        params.expt_name = "idk4_2_alpha100e"
        params.dataset = "/data/lisa/data/cbeckham/textures_v2_brown500.h5"
        params.batch_size = 8
        params.epochs = 1000
        params.alpha = 100.
        params.reconstruction_only = False
        params.lr = 1e-3
        print_params()
        dopt = Adam(lr=params.lr, beta_1=params.beta_1)
        from architectures import default_models
        # Define the generator
        unet = default_models.g_unet(in_ch=params.a_ch,
                             out_ch=params.b_ch,
                             nf=64,
                             num_padded_conv=0,
                             batch_size=params.batch_size,
                             is_grayscale=params.is_b_grayscale)
        # Define the discriminator
        d = m.discriminator(params.a_ch, params.b_ch, 1, depths=[1,2,4], bn=True, opt=dopt) #[1,2,4,8,16,24,32,40] 
        if params.continue_train:
            print "loading weights..."
            load_weights(unet, d, log_dir=params.log_dir, expt_name=params.expt_name)
        models = model_creation(d, unet, params)
        it_train, it_val = get_iterators()
        if mode == "train":
            train(models, it_train, it_val, params)

    def idk4f(mode,seed):
        np.random.seed(seed)
        # override params here
        params.expt_name = "idk4_2_alpha100f_noda"
        params.dataset = "/data/lisa/data/cbeckham/textures_v2_brown500.h5"
        params.batch_size = 8
        params.epochs = 1000
        params.alpha = 100.
        params.reconstruction_only = False
        params.lr = 1e-3
        print_params()
        dopt = Adam(lr=params.lr, beta_1=params.beta_1)
        from architectures import default_models
        # Define the generator
        unet = default_models.g_unet(in_ch=params.a_ch,
                             out_ch=params.b_ch,
                             nf=64,
                             num_padded_conv=2,
                             batch_size=params.batch_size,
                             is_grayscale=params.is_b_grayscale)
        # Define the discriminator
        d = m.discriminator(params.a_ch, params.b_ch, 1, depths=[1,2,4,8,16,32,64,128], bn=True, opt=dopt) #[1,2,4,8,16,24,32,40] 
        if params.continue_train:
            print "loading weights..."
            load_weights(unet, d, log_dir=params.log_dir, expt_name=params.expt_name)
        models = model_creation(d, unet, params)
        it_train, it_val = get_iterators(da=False)
        if mode == "train":
            train(models, it_train, it_val, params)


            
    def gmaps1(mode,seed):
        np.random.seed(seed)
        # override params here
        params.expt_name = "gmaps1"
        params.dataset = "/data/lisatmp4/beckhamc/pix2pix-tensorflow/maps/maps.h5"
        params.batch_size = 16
        params.epochs = 1000
        params.alpha = 100.
        #params.reconstruction_only = True
        params.lr = 1e-3
        params.is_a_binary, params.is_b_binary = False, False
        params.a_ch, params.b_ch = 3, 3
        print_params()
        dopt = Adam(lr=params.lr, beta_1=params.beta_1)
        from architectures import default_models
        # Define the generator
        unet = default_models.g_unet(in_ch=params.a_ch,
                             out_ch=params.b_ch,
                             nf=64,
                             batch_size=params.batch_size,
                             is_binary=params.is_b_binary)
        # Define the discriminator
        d = m.discriminator(params.a_ch, params.b_ch, 32, bn=False, opt=dopt)
        if params.continue_train:
            print "loading weights..."
            load_weights(unet, d, log_dir=params.log_dir, expt_name=params.expt_name)
        models = model_creation(d, unet, params)
        it_train, it_val = get_iterators()
        if mode == "train":
            train(models, it_train, it_val, params)





            
    locals()[ sys.argv[1] ]( sys.argv[2], int(sys.argv[3]) )
