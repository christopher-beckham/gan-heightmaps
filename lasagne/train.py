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
sys.path.append("..") # some important shit we need to import
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from time import time
from nolearn.lasagne.visualize import draw_to_file
import nolearn
from keras_ports import ReduceLROnPlateau
import pickle

def Convolution(layer, f, k=3, s=2, border_mode='same', **kwargs):
    """Convenience method for Convolutions."""
    """
    if KERAS_2:
        return Convolution2D(f,
                             kernel_size=(k, k),
                             padding=border_mode,
                             strides=(s, s),
                             **kwargs)
    else:
        return Convolution2D(f, k, k, border_mode=border_mode,
                             subsample=(s, s),
                             **kwargs)
    """
    
    return Conv2DLayer(layer, num_filters=f, filter_size=(k,k), stride=(s,s), pad=border_mode, nonlinearity=linear)

def Deconvolution(layer, f, k=2, s=2, **kwargs):
    """Convenience method for Transposed Convolutions."""
    """
    if KERAS_2:
        return Conv2DTranspose(f,
                               kernel_size=(k, k),
                               strides=(s, s),
                               data_format=K.image_data_format(),
                               **kwargs)
    else:
        return Deconvolution2D(f, k, k, output_shape=output_shape,
                               subsample=(s, s), **kwargs)
    """
    return Deconv2DLayer(layer, num_filters=f, filter_size=(k,k), stride=(s,s), nonlinearity=linear)

def concatenate_layers(layers, **kwargs):
    return ConcatLayer(layers, axis=1)

def g_unet(nf=64, act=tanh, num_repeats=0):
    print num_repeats
    def padded_conv(nf, x):
        x = Convolution(x, nf,s=1,k=3)
        x = BatchNormLayer(x)
        x = NonlinearityLayer(x, nonlinearity=leaky_rectify)
        return x
    i = InputLayer((None, 1, 512, 512))
    # in_ch x 512 x 512
    conv1 = Convolution(i, nf)
    conv1 = BatchNormLayer(conv1)
    x = NonlinearityLayer(conv1, nonlinearity=leaky_rectify)
    for r in range(num_repeats):
        x = padded_conv(nf, x)    
    # nf x 256 x 256
    conv2 = Convolution(x, nf * 2)
    conv2 = BatchNormLayer(conv2)
    x = NonlinearityLayer(conv2, nonlinearity=leaky_rectify)
    for r in range(num_repeats):
        x = padded_conv(nf*2, x)
    # nf*2 x 128 x 128
    conv3 = Convolution(x, nf * 4)
    conv3 = BatchNormLayer(conv3)
    x = NonlinearityLayer(conv3, nonlinearity=leaky_rectify)
    for r in range(num_repeats):
        x = padded_conv(nf*4, x)
    # nf*4 x 64 x 64
    conv4 = Convolution(x, nf * 8)
    conv4 = BatchNormLayer(conv4)
    x = NonlinearityLayer(conv4, nonlinearity=leaky_rectify)
    for r in range(num_repeats):
        x = padded_conv(nf*8, x)
    # nf*8 x 32 x 32
    conv5 = Convolution(x, nf * 8)
    conv5 = BatchNormLayer(conv5)
    x = NonlinearityLayer(conv5, nonlinearity=leaky_rectify)
    for r in range(num_repeats):
        x = padded_conv(nf*8, x)
    # nf*8 x 16 x 16
    conv6 = Convolution(x, nf * 8)
    conv6 = BatchNormLayer(conv6)
    x = NonlinearityLayer(conv6, nonlinearity=leaky_rectify)
    for r in range(num_repeats):
        x = padded_conv(nf*8, x)
    # nf*8 x 8 x 8
    conv7 = Convolution(x, nf * 8)
    conv7 = BatchNormLayer(conv7)
    x = NonlinearityLayer(conv7, nonlinearity=leaky_rectify)
    for r in range(num_repeats):
        x = padded_conv(nf*8, x)
    # nf*8 x 4 x 4
    conv8 = Convolution(x, nf * 8)
    conv8 = BatchNormLayer(conv8)
    x = NonlinearityLayer(conv8, nonlinearity=leaky_rectify)
    for r in range(num_repeats):
        x = padded_conv(nf*8, x)
    # nf*8 x 2 x 2
    conv9 = Convolution(x, nf * 8, k=2, s=1, border_mode='valid')
    conv9 = BatchNormLayer(conv9)
    x = NonlinearityLayer(conv9, nonlinearity=leaky_rectify)
    # nf*8 x 1 x 1  
    dconv1 = Deconvolution(x, nf * 8,
                           k=2, s=1)
    dconv1 = BatchNormLayer(dconv1)
    x = concatenate_layers([dconv1, conv8])
    x = NonlinearityLayer(x, nonlinearity=leaky_rectify)
    for r in range(num_repeats):
        x = padded_conv(nf*8, x)
    # nf*(8 + 8) x 2 x 2
    dconv2 = Deconvolution(x, nf * 8)
    dconv2 = BatchNormLayer(dconv2)
    x = concatenate_layers([dconv2, conv7])
    x = NonlinearityLayer(x, leaky_rectify)
    for r in range(num_repeats):
        x = padded_conv(nf*8, x)
    # nf*(8 + 8) x 4 x 4
    dconv3 = Deconvolution(x, nf * 8)
    dconv3 = BatchNormLayer(dconv3)
    #dconv3 = DropoutLayer(dconv3, 0.5)
    x = concatenate_layers([dconv3, conv6])
    x = NonlinearityLayer(x, leaky_rectify)
    for r in range(num_repeats):
        x = padded_conv(nf*8, x)
    # nf*(8 + 8) x 8 x 8
    dconv4 = Deconvolution(x, nf * 8)
    dconv4 = BatchNormLayer(dconv4)
    x = concatenate_layers([dconv4, conv5])
    x = NonlinearityLayer(x, leaky_rectify)
    for r in range(num_repeats):
        x = padded_conv(nf*8, x)
    # nf*(8 + 8) x 16 x 16
    dconv5 = Deconvolution(x, nf * 8)
    dconv5 = BatchNormLayer(dconv5)
    x = concatenate_layers([dconv5, conv4])
    x = NonlinearityLayer(x, leaky_rectify)
    for r in range(num_repeats):
        x = padded_conv(nf*8, x)
    # nf*(8 + 8) x 32 x 32
    dconv6 = Deconvolution(x, nf * 4)
    dconv6 = BatchNormLayer(dconv6)
    x = concatenate_layers([dconv6, conv3])
    x = NonlinearityLayer(x, leaky_rectify)
    for r in range(num_repeats):
        x = padded_conv(nf*4, x)    
    # nf*(4 + 4) x 64 x 64
    dconv7 = Deconvolution(x, nf * 2)
    dconv7 = BatchNormLayer(dconv7)
    x = concatenate_layers([dconv7, conv2])
    x = NonlinearityLayer(x, leaky_rectify)
    for r in range(num_repeats):
        x = padded_conv(nf*2, x)
    # nf*(2 + 2) x 128 x 128
    dconv8 = Deconvolution(x, nf)
    dconv8 = BatchNormLayer(dconv8)
    x = concatenate_layers([dconv8, conv1])
    x = NonlinearityLayer(x, leaky_rectify)
    for r in range(num_repeats):
        x = padded_conv(nf, x)
    # nf*(1 + 1) x 256 x 256
    dconv9 = Deconvolution(x, 3)
    # out_ch x 512 x 512
    #act = 'sigmoid' if is_binary else 'tanh'
    out = NonlinearityLayer(dconv9, act)
    return out

def discriminator(nf=32, act=sigmoid, mul_factor=[1,2,4,8], num_repeats=0, bn=False):
    i_a = InputLayer((None, 1, 512, 512))
    i_b = InputLayer((None, 3, 512, 512))
    i = concatenate_layers([i_a, i_b])
    x = i
    for m in mul_factor:
        for r in range(num_repeats+1):
            x = Convolution(x, nf*m, s=2 if r == 0 else 1)
            x = NonlinearityLayer(x, leaky_rectify)
            if bn:
                x = BatchNormLayer(x)
    x = Convolution(x, 1)
    out = NonlinearityLayer(x, act)
    # 1 x 16 x 16
    return {"inputs": [i_a, i_b], "out":out}

def print_network(l_out):
    for layer in get_all_layers(l_out):
        print layer, layer.output_shape, "" if not hasattr(layer, 'nonlinearity') else layer.nonlinearity
    print "num of learnable params:", count_params(layer, trainable=True)

    
def prepare(gen_params, disc_params):

    l_gen = g_unet(**gen_params)
    dd_disc = discriminator(**disc_params)

    # print stuff
    print_network(l_gen)
    print_network(dd_disc["out"])

    X = T.tensor4('X') # this is the input heightmap (bs, 1, 512, 512)
    Y = T.tensor4('Y') # this is the ground truth texture (bs, 3, 512, 512)
    # this is the output of the discriminator for real (x,y)
    disc_out_real = get_output(
        dd_disc["out"],
        { dd_disc["inputs"][0]: X, dd_disc["inputs"][1]: Y }
    )
    # this is the output of the discriminator for fake (x, y')
    gen_out = get_output(l_gen, X)
    disc_out_fake = get_output(
        dd_disc["out"],
        { dd_disc["inputs"][0]: X, dd_disc["inputs"][1]: gen_out }
    )

    # want to fool discriminator that this is real, e.g. == 1

    if params.lsgan:
        adv_loss = squared_error
    else:
        adv_loss = binary_crossentropy
    
    gen_loss = adv_loss(disc_out_fake, 1.).mean()
    assert params.reconstruction in ['l1', 'l2']
    if params.reconstruction == 'l2':
        recon_loss = squared_error(gen_out, Y).mean()
    else:
        print "using l1"
        recon_loss = T.abs_(gen_out-Y).mean()
    if not params.reconstruction_only:
        gen_total_loss = gen_loss + params.alpha*recon_loss
    else:
        gen_total_loss = recon_loss

    # discriminator wants to say real stuff is real, and fake stuff is fake...
    disc_loss = adv_loss(disc_out_real, 1.).mean() + adv_loss(disc_out_fake, 0.).mean()

    gen_params = get_all_params(l_gen, trainable=True)
    disc_params = get_all_params(dd_disc["out"], trainable=True)

    assert params.opt in ['adam', 'rmsprop']
    if params.opt == 'adam':
        opt = adam
    else:
        opt = rmsprop
    from lasagne.utils import floatX
    lr = theano.shared(floatX(params.lr))
    updates = opt(gen_total_loss, gen_params, learning_rate=lr)
    if not params.reconstruction_only:
        updates.update(opt(disc_loss, disc_params, learning_rate=lr))
    cb = ReduceLROnPlateau(lr,verbose=1)
    train_fn = theano.function([X,Y], [gen_loss, recon_loss, disc_loss], updates=updates)
    gen_fn = theano.function([X], gen_out)
    
    return {"train_fn":train_fn, "gen_fn":gen_fn, "l_gen":l_gen, "l_disc":dd_disc["out"], "cb":cb}

# --------------------

from util.data import TwoImageIterator, iterate_hdf5, Hdf5Iterator
from util.util import MyDict, log, save_weights, load_weights, load_losses, create_expt_dir, convert_to_rgb, compose_imgs

def get_iterators(dataset, batch_size, is_a_grayscale, is_b_grayscale, da=True):
    dataset = h5py.File(dataset,"r")
    if da:
        imgen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, rotation_range=360, fill_mode="reflect")
    else:
        imgen = ImageDataGenerator()
    it_train = Hdf5Iterator(dataset['xt'], dataset['yt'], batch_size, imgen, is_a_grayscale=is_a_grayscale, is_b_grayscale=is_b_grayscale)
    it_val = Hdf5Iterator(dataset['xv'], dataset['yv'], batch_size, imgen, is_a_grayscale=is_a_grayscale, is_b_grayscale=is_b_grayscale)
    return it_train, it_val

def plot_grid(out_filename, itr, out_fn, is_a_grayscale, is_b_grayscale, N=4):
    plt.figure(figsize=(10, 6))
    for i in range(N*N):
        a, b = itr.next()
        if out_fn != None:
            bp = out_fn(a)
        else:
            bp = b
        img = compose_imgs(a[0], bp[0], is_a_grayscale=is_a_grayscale, is_b_grayscale=is_b_grayscale)
        plt.subplot(N, N, i+1)
        plt.imshow(img)
        plt.axis('off')
    plt.savefig(out_filename)
    plt.clf()
    # Make sure all the figures are closed.
    plt.close('all')

def save_model(l_gen, l_disc, filename):
    with open(filename, "wb") as g:
        pickle.dump( (get_all_param_values(l_gen), get_all_param_values(l_disc)), g, pickle.HIGHEST_PROTOCOL )

def load_model(l_gen, l_disc, filename):
    with open(filename) as g:
        wts = pickle.load(g)
        set_all_param_values(l_gen, wts[0])
        set_all_param_values(l_disc, wts[1])
               
def train(net_cfg, it_train, it_val, out_dir, model_dir=None, save_every=10, resume=None):
    train_fn, gen_fn = net_cfg["train_fn"], net_cfg["gen_fn"]
    l_gen, l_disc = net_cfg["l_gen"], net_cfg["l_disc"]
    cb = net_cfg["cb"] # callback for LR plateau
    f = open("%s/results.txt" % out_dir, "wb" if resume==None else "a")
    if resume == None:
        f.write("epoch,gen,recon,disc,lr,time\n"); f.flush()
    losses = {'gen':[], 'recon':[], 'disc':[]}
    for e in range(params.num_epochs):
        t0 = time()
        gen_losses = []
        recon_losses = []
        disc_losses = []
        for b in range(it_train.N // params.batch_size):
            X_batch, Y_batch = it_train.next()
            gen_loss, recon_loss, disc_loss = train_fn(X_batch,Y_batch)
            gen_losses.append(gen_loss)
            recon_losses.append(recon_loss)
            disc_losses.append(disc_loss)
        losses['gen'].append(np.mean(gen_losses))
        losses['recon'].append(np.mean(recon_losses))
        if params.reduce_on_plateau:
            cb.on_epoch_end(np.mean(recon_losses), e+1)
        losses['disc'].append(np.mean(disc_losses))
        out_str = "%i,%f,%f,%f,%f,%f" % (e+1, losses['gen'][-1], losses['recon'][-1], losses['disc'][-1], cb.learning_rate.get_value(), time()-t0)
        print out_str
        f.write("%s\n" % out_str); f.flush()
        plot_grid("%s/out_%i.png" % (out_dir,e+1), it_val, gen_fn, is_a_grayscale=params.is_a_grayscale, is_b_grayscale=params.is_b_grayscale)
        if model_dir != None and (e+1) % save_every == 0:
            save_model(l_gen, l_disc, "%s/%i.model" % (model_dir, e+1)) 

def generate_imgs(itr, gen_fn, num_batches, out_dir, is_grayscale=False):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    from skimage.io import imsave
    ctr = 0
    for n in range(num_batches):
        this_x, this_y = itr.next()
        if gen_fn != None:
            pred_y = gen_fn(this_x)
        else:
            pred_y = this_y
        for i in range(pred_y.shape[0]):
            img = convert_to_rgb(pred_y[i], is_grayscale=is_grayscale)
            import pdb
            pdb.set_trace()
            imsave(fname="%s/%i.texture.png" % (out_dir, ctr), arr=img)
            imsave(fname="%s/%i.hm.png" % (out_dir, ctr), arr=this_x[i][0])
            ctr += 1
            
params = MyDict({
    'lr': 1e-3,
    'dataset': '/data/lisa/data/cbeckham/textures_v2_brown500.h5',
    'batch_size': 4,
    'num_epochs': 500,
    'is_a_grayscale': True,
    'is_b_grayscale': False,
    'da':True,
    'alpha':100,
    'reconstruction':'l2',
    'reconstruction_only':False,
    'lsgan': False,
    'opt':'adam',
    'reduce_on_plateau':False
})

if __name__ == '__main__':
    desert_h5 = "/data/lisa/data/cbeckham/textures_v2_brown500.h5"
    def exp1():
        it_train, it_val = get_iterators(desert_h5,
                batch_size=params.batch_size, is_a_grayscale=params.is_a_grayscale, is_b_grayscale=params.is_a_grayscale, da=params.da)
        params.alpha = 1.
        name = "exp1_d1_alpha1"
        out_dir = "output/%s" % name
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        net_cfg = prepare(gen_params={'nf':64, 'act':tanh}, disc_params={'nf':32, 'bn':False, 'mul_factor':[1,2,4,8,16,32]})
        draw_to_file(get_all_layers(net_cfg["l_gen"]), "%s/gen.png" % out_dir, verbose=True)
        draw_to_file(get_all_layers(net_cfg["l_disc"]), "%s/disc.png" % out_dir, verbose=True)
        train(net_cfg, it_train=it_train, it_val=it_val, out_dir=out_dir)

    def exp1_alpha50():
        it_train, it_val = get_iterators(desert_h5,
                batch_size=params.batch_size, is_a_grayscale=params.is_a_grayscale, is_b_grayscale=params.is_a_grayscale, da=params.da)
        params.alpha = 50.
        name = "exp1_d1_alpha50"
        out_dir = "output/%s" % name
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        net_cfg = prepare(gen_params={'nf':64, 'act':tanh}, disc_params={'nf':32, 'bn':False, 'mul_factor':[1,2,4,8,16,32]})
        draw_to_file(get_all_layers(net_cfg["l_gen"]), "%s/gen.png" % out_dir, verbose=True)
        draw_to_file(get_all_layers(net_cfg["l_disc"]), "%s/disc.png" % out_dir, verbose=True)
        train(net_cfg, it_train=it_train, it_val=it_val, out_dir=out_dir)

    def exp1_alpha50_l1():
        it_train, it_val = get_iterators(desert_h5,
                batch_size=params.batch_size, is_a_grayscale=params.is_a_grayscale, is_b_grayscale=params.is_a_grayscale, da=params.da)
        params.alpha = 50.
        params.reconstruction = 'l1'
        name = "exp1_d1_alpha50_l1"
        out_dir = "output/%s" % name
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        net_cfg = prepare(gen_params={'nf':64, 'act':tanh}, disc_params={'nf':32, 'bn':False, 'mul_factor':[1,2,4,8,16,32]})
        draw_to_file(get_all_layers(net_cfg["l_gen"]), "%s/gen.png" % out_dir, verbose=True)
        draw_to_file(get_all_layers(net_cfg["l_disc"]), "%s/disc.png" % out_dir, verbose=True)
        train(net_cfg, it_train=it_train, it_val=it_val, out_dir=out_dir)

    def exp1_alpha5_l1():
        it_train, it_val = get_iterators(desert_h5,
                batch_size=params.batch_size, is_a_grayscale=params.is_a_grayscale, is_b_grayscale=params.is_a_grayscale, da=params.da)
        params.alpha = 5.
        params.reconstruction = 'l1'
        name = "exp1_d1_alpha5"
        out_dir = "output/%s" % name
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        net_cfg = prepare(gen_params={'nf':64, 'act':tanh}, disc_params={'nf':32, 'bn':False, 'mul_factor':[1,2,4,8,16,32]})
        draw_to_file(get_all_layers(net_cfg["l_gen"]), "%s/gen.png" % out_dir, verbose=True)
        draw_to_file(get_all_layers(net_cfg["l_disc"]), "%s/disc.png" % out_dir, verbose=True)
        train(net_cfg, it_train=it_train, it_val=it_val, out_dir=out_dir)

    def exp1_alpha5_l1_lsgan():
        it_train, it_val = get_iterators(desert_h5,
                batch_size=params.batch_size, is_a_grayscale=params.is_a_grayscale, is_b_grayscale=params.is_a_grayscale, da=params.da)
        params.alpha = 5.
        params.reconstruction = 'l1'
        params.opt = 'rmsprop'
        params.lsgan = True
        params.lr = 1e-4
        name = "exp1_d1_alpha5_lsgan_nod5"
        out_dir = "output/%s" % name
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        net_cfg = prepare(gen_params={'nf':64, 'act':tanh}, disc_params={'nf':32, 'bn':False, 'mul_factor':[1,2,4,8,16,32]})
        draw_to_file(get_all_layers(net_cfg["l_gen"]), "%s/gen.png" % out_dir, verbose=True)
        draw_to_file(get_all_layers(net_cfg["l_disc"]), "%s/disc.png" % out_dir, verbose=True)
        train(net_cfg, it_train=it_train, it_val=it_val, out_dir=out_dir)

    def exp1_alpha5_l1_lsgan_bigd1():
        it_train, it_val = get_iterators(desert_h5,
                batch_size=params.batch_size, is_a_grayscale=params.is_a_grayscale, is_b_grayscale=params.is_a_grayscale, da=params.da)
        params.alpha = 5.
        params.reconstruction = 'l1'
        params.opt = 'rmsprop'
        params.lsgan = True
        params.lr = 1e-4
        name = "exp1_d1_alpha5_lsgan_nod5_bigd1"
        out_dir = "output/%s" % name
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        net_cfg = prepare(gen_params={'nf':64, 'act':tanh}, disc_params={'nf':8, 'bn':False, 'mul_factor':[1,2,4,8,16,32,64,128]})
        draw_to_file(get_all_layers(net_cfg["l_gen"]), "%s/gen.png" % out_dir, verbose=True)
        draw_to_file(get_all_layers(net_cfg["l_disc"]), "%s/disc.png" % out_dir, verbose=True)
        train(net_cfg, it_train=it_train, it_val=it_val, out_dir=out_dir)


    def exp1_alpha1_l1_lsgan_bigd1():
        it_train, it_val = get_iterators(desert_h5,
                batch_size=params.batch_size, is_a_grayscale=params.is_a_grayscale, is_b_grayscale=params.is_a_grayscale, da=params.da)
        params.alpha = 1.
        params.reconstruction = 'l1'
        params.opt = 'rmsprop'
        params.lsgan = True
        params.lr = 1e-4
        name = "exp1_d1_alpha1_lsgan_nod5_bigd1"
        out_dir = "output/%s" % name
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        net_cfg = prepare(gen_params={'nf':64, 'act':tanh}, disc_params={'nf':8, 'bn':False, 'mul_factor':[1,2,4,8,16,32,64,128]})
        draw_to_file(get_all_layers(net_cfg["l_gen"]), "%s/gen.png" % out_dir, verbose=True)
        draw_to_file(get_all_layers(net_cfg["l_disc"]), "%s/disc.png" % out_dir, verbose=True)
        train(net_cfg, it_train=it_train, it_val=it_val, out_dir=out_dir)

    def exp1_alpha1_l1_lsgan_bigd1_dnr1_patchg(mode):
        assert mode in ["train", "test"]
        it_train, it_val = get_iterators(desert_h5,
                batch_size=params.batch_size, is_a_grayscale=params.is_a_grayscale, is_b_grayscale=params.is_a_grayscale, da=params.da)
        params.alpha = 100.
        params.reconstruction = 'l1'
        params.opt = 'rmsprop'
        params.lsgan = True
        params.lr = 1e-4
        name = "exp1_d1_alpha100_lsgan_nod5_dbn3_patchg_repeat"
        out_dir = "output/%s" % name
        model_dir = "models/%s" % name
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        net_cfg = prepare(gen_params={'nf':64, 'act':tanh, 'num_repeats':1}, disc_params={'nf':64, 'bn':True, 'num_repeats': 0, 'act':linear, 'mul_factor':[1,2,4,8]})
        if mode == "train":
            draw_to_file(get_all_layers(net_cfg["l_gen"]), "%s/gen.png" % out_dir, verbose=True)
            draw_to_file(get_all_layers(net_cfg["l_disc"]), "%s/disc.png" % out_dir, verbose=True)
            train(net_cfg, it_train=it_train, it_val=it_val, out_dir=out_dir, model_dir=model_dir)
        else:
            print "testing..."
            load_model(net_cfg["l_gen"], net_cfg["l_disc"], "models/%s/190.model" % name)
            custom_h5 = h5py.File("../dcgan/images_512/gan-heightmap-ld1000-b-discbn_i1ls_rmspropd_weakd/dump.h5")
            custom_itr = Hdf5Iterator(custom_h5['xt'], custom_h5['yt'], bs=4,
                                      imgen=ImageDataGenerator(), is_a_grayscale=params.is_a_grayscale, is_b_grayscale=params.is_b_grayscale, is_uint8=False)
            #print "generating imgs..."
            generate_imgs(custom_itr, net_cfg["gen_fn"], num_batches=10, out_dir="images_512/%s" % name)
            #import pdb
            #pdb.set_trace()
            #plot_grid("images_512/%s/grid.png" % name, custom_itr, net_cfg["gen_fn"], True, False)

    def test(mode):
        assert mode in ["test"]
        it_train, it_val = get_iterators(desert_h5,
                batch_size=params.batch_size, is_a_grayscale=params.is_a_grayscale, is_b_grayscale=params.is_a_grayscale, da=params.da)
        plot_grid("grid_ground_truth.png", it_train, None, params.is_a_grayscale, params.is_b_grayscale)
            
    def exp1_alpha1_l1_lsgan_bigd1_dnr1_recon_only():
        it_train, it_val = get_iterators(desert_h5,
                batch_size=params.batch_size, is_a_grayscale=params.is_a_grayscale, is_b_grayscale=params.is_a_grayscale, da=params.da)
        params.alpha = 100.
        params.reconstruction = 'l1'
        params.opt = 'rmsprop'
        params.lsgan = True
        params.lr = 1e-4
        params.reconstruction_only = True
        name = "exp1_d1_alpha100_lsgan_nod5_dbn3_recon-only"
        out_dir = "output/%s" % name
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        net_cfg = prepare(gen_params={'nf':64, 'act':tanh, 'num_repeats':1}, disc_params={'nf':8, 'bn':True, 'num_repeats': 0, 'act':linear, 'mul_factor':[1,2,4,8,16,32,64,128]})
        draw_to_file(get_all_layers(net_cfg["l_gen"]), "%s/gen.png" % out_dir, verbose=True)
        draw_to_file(get_all_layers(net_cfg["l_disc"]), "%s/disc.png" % out_dir, verbose=True)
        train(net_cfg, it_train=it_train, it_val=it_val, out_dir=out_dir)


        
    locals()[ sys.argv[1] ]( sys.argv[2] )
"""
if __name__ == '__main__':
    g = g_unet()
    d = discriminator()
    print_network(g)
    print_network(d)
"""
