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
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from time import time
import nolearn
from keras_ports import ReduceLROnPlateau
import pickle
import sys
sys.path.append("..")
from util.data import iterate_hdf5, Hdf5Iterator
from util.util import convert_to_rgb, compose_imgs

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
    
class Pix2Pix():
    def _print_network(self,l_out):
        for layer in get_all_layers(l_out):
            print layer, layer.output_shape, "" if not hasattr(layer, 'nonlinearity') else layer.nonlinearity
        print "# learnable params:", count_params(layer, trainable=True)
    def __init__(self,
                 gen_fn_dcgan, disc_fn_dcgan,
                 gen_params_dcgan, disc_params_dcgan,
                 gen_fn_p2p, disc_fn_p2p,
                 gen_params_p2p, disc_params_p2p,
                 in_shp, latent_dim, is_a_grayscale, is_b_grayscale,
                 alpha=100, lr=1e-4, opt=adam, opt_args={'learning_rate':theano.shared(floatX(1e-3))},
                 reconstruction='l1', sampler=np.random.rand, reconstruction_only=False, lsgan=False, verbose=True):
        self.is_a_grayscale = is_a_grayscale
        self.is_b_grayscale = is_b_grayscale
        self.sampler = sampler
        self.verbose = verbose
        # get the networks for the dcgan network
        dcgan_gen = gen_fn_dcgan(latent_dim, is_a_grayscale, **gen_params_dcgan)
        dcgan_disc = disc_fn_dcgan(is_a_grayscale, **disc_params_dcgan)
        # get the networks for the p2p network
        p2p_gen = gen_fn_p2p(in_shp, is_a_grayscale, is_b_grayscale, **gen_params_p2p)
        p2p_disc = disc_fn_p2p(in_shp, is_a_grayscale, is_b_grayscale, **disc_params_p2p)
        if verbose:
            self._print_network(p2p_gen)
            self._print_network(p2p_disc["out"])
        Z = T.fmatrix('Z') # noise var
        X = T.tensor4('X') # A
        Y = T.tensor4('Y') # B
        # construct theano stuff for dcgan gen/disc
        dcgan = {}
        dcgan['gen_out'] = get_output(dcgan_gen, Z) # G(z)
        dcgan['disc_out_real'] = get_output(dcgan_disc, X) # D(x)
        dcgan['disc_out_fake'] = get_output(dcgan_disc, dcgan['gen_out']) # D(G(z))
        # construct theano stuff for the p2p gen/disc
        p2p = {}
        p2p['disc_out_real'] = get_output(p2p_disc["out"], { p2p_disc["inputs"][0]: X, p2p_disc["inputs"][1]: Y }) # D(X,Y)
        p2p['gen_out'] = get_output(p2p_gen, X)
        p2p['disc_out_fake'] = get_output(p2p_disc["out"], { p2p_disc["inputs"][0]: X, p2p_disc["inputs"][1]: p2p['gen_out'] }) # D(X, X_to_y(X))
        if lsgan:
            adv_loss = squared_error
        else:
            adv_loss = binary_crossentropy
        # dcgan loss definitions
        gen_loss_dcgan = adv_loss(dcgan['disc_out_fake'], 1.).mean()
        disc_loss_dcgan = adv_loss(dcgan['disc_out_real'], 1.).mean() + adv_loss(dcgan['disc_out_fake'], 0.).mean()
        # p2p loss definitions
        gen_loss_p2p = adv_loss(dcgan['disc_out_fake'], 1.).mean()
        assert reconstruction in ['l1', 'l2']
        if reconstruction == 'l2':
            recon_loss = squared_error(p2p['gen_out'], Y).mean()
        else:
            recon_loss = T.abs_(p2p['gen_out']-Y).mean()
        if not reconstruction_only:
            gen_total_loss_p2p = gen_loss_p2p + alpha*recon_loss
        else:
            #log("GAN disabled, using only pixel-wise reconstruction loss...")
            gen_total_loss_p2p = recon_loss
        disc_loss_p2p = adv_loss(p2p['disc_out_real'], 1.).mean() + adv_loss(p2p['disc_out_fake'], 0.).mean()
        # dcgan params
        gen_params_dcgan = get_all_params(dcgan_gen, trainable=True)
        disc_params_dcgan = get_all_params(dcgan_disc, trainable=True)
        # pix2pix params
        gen_params_p2p = get_all_params(p2p_gen, trainable=True)
        disc_params_p2p = get_all_params(p2p_disc["out"], trainable=True)
        # --------------------
        #from lasagne.utils import floatX
        #lr = theano.shared(floatX(lr))
        updates = opt(gen_loss_dcgan, gen_params_dcgan, **opt_args) # update dcgan generator
        updates.update(opt(disc_loss_dcgan, disc_params_dcgan, **opt_args)) # update dcgan discriminator
        updates.update(opt(gen_total_loss_p2p, gen_params_p2p, **opt_args)) # update p2p generator
        if not reconstruction_only:
            updates.update(opt(disc_loss_p2p, disc_params_p2p, **opt_args)) # update p2p discriminator
        train_fn = theano.function([Z,X,Y], [gen_loss_dcgan, disc_loss_dcgan, gen_loss_p2p, recon_loss, disc_loss_p2p], updates=updates)
        loss_fn = theano.function([Z,X,Y], [gen_loss_dcgan, disc_loss_dcgan, gen_loss_p2p, recon_loss, disc_loss_p2p])
        gen_fn = theano.function([X], p2p['gen_out'])
        z_fn = theano.function([Z], dcgan['gen_out'])
        self.train_fn = train_fn
        self.loss_fn = loss_fn
        self.gen_fn = gen_fn
        self.p2p_gen = p2p_gen
        #self.l_disc = p2p_disc["out"]
        self.lr = opt_args['learning_rate']
    def save_model(self, filename):
        with open(filename, "wb") as g:
            pickle.dump( (get_all_param_values(self.p2p_gen), get_all_param_values(self.p2p_disc)), g, pickle.HIGHEST_PROTOCOL )
    def load_model(self, filename):
        with open(filename) as g:
            wts = pickle.load(g)
            set_all_param_values(self.p2p_gen, wts[0])
            set_all_param_values(self.p2p_disc, wts[1])            
    def train(self, it_train, it_val, batch_size, num_epochs, out_dir, model_dir=None, save_every=1, resume=None, reduce_on_plateau=False):
        def _loop(fn, itr):
            gen_losses, recon_losses, disc_losses = [], [], []
            for b in range(itr.N // batch_size):
                X_batch, Y_batch = it_train.next()
                #print X_batch.shape, Y_batch.shape
                gen_loss, recon_loss, disc_loss = fn(X_batch,Y_batch)
                gen_losses.append(gen_loss)
                recon_losses.append(recon_loss)
                disc_losses.append(disc_loss)
            return np.mean(gen_losses), np.mean(recon_losses), np.mean(disc_losses)            
        header = ["epoch","train_gen","train_recon","train_disc","valid_gen","valid_recon","valid_disc","lr","time"]
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        if model_dir != None and not os.path.exists(model_dir):
            os.makedirs(model_dir)
        if self.verbose:
            try:
                from nolearn.lasagne.visualize import draw_to_file
                draw_to_file(get_all_layers(self.p2p_gen), "%s/gen.png" % out_dir, verbose=True)
                draw_to_file(get_all_layers(self.p2p_disc), "%s/disc.png" % out_dir, verbose=True)
            except:
                pass
        f = open("%s/results.txt" % out_dir, "wb" if resume==None else "a")
        if resume == None:
            f.write(",".join(header)+"\n"); f.flush()
        else:
            if self.verbose:
                print "loading weights from: %s" % resume
            self.load_model(resume)
        train_losses = {'gen':[], 'recon':[], 'disc':[]}
        valid_losses = {'gen':[], 'recon':[], 'disc':[]}
        cb = ReduceLROnPlateau(self.lr,verbose=self.verbose)
        for e in range(num_epochs):
            t0 = time()
            # training
            a,b,c = _loop(self.train_fn, it_train)
            train_losses['gen'].append(a)
            train_losses['recon'].append(b)
            train_losses['disc'].append(c)
            if reduce_on_plateau:
                cb.on_epoch_end(np.mean(recon_losses), e+1)
            # validation
            a,b,c = _loop(self.loss_fn, it_val)
            valid_losses['gen'].append(a)
            valid_losses['recon'].append(b)
            valid_losses['disc'].append(c)
            out_str = "%i,%f,%f,%f,%f,%f,%f,%f,%f" % \
                      (e+1,
                       train_losses['gen'][-1],
                       train_losses['recon'][-1],
                       train_losses['disc'][-1],
                       valid_losses['gen'][-1],
                       valid_losses['recon'][-1],
                       valid_losses['disc'][-1],
                       cb.learning_rate.get_value(),
                       time()-t0)
            print out_str
            f.write("%s\n" % out_str); f.flush()
            # plot an NxN grid of [A, predict(A)]
            plot_grid("%s/out_%i.png" % (out_dir,e+1), it_val, self.gen_fn, is_a_grayscale=self.is_a_grayscale, is_b_grayscale=self.is_b_grayscale)
            # plot big pictures of predict(A) in the valid set
            self.generate_imgs(it_train, 1, "%s/dump_train" % out_dir)
            self.generate_imgs(it_val, 1, "%s/dump_valid" % out_dir)
            if model_dir != None and e % save_every == 0:
                self.save_model("%s/%i.model" % (model_dir, e+1))
    def generate_imgs(self, itr, num_batches, out_dir, dont_predict=False):
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        from skimage.io import imsave
        ctr = 0
        for n in range(num_batches):
            this_x, this_y = itr.next()
            if dont_predict:
                pred_y = this_y
            else:
                pred_y = self.gen_fn(this_x)
            for i in range(pred_y.shape[0]):
                this_x_processed = convert_to_rgb(this_x[i], is_grayscale=self.is_a_grayscale)
                pred_y_processed = convert_to_rgb(pred_y[i], is_grayscale=self.is_b_grayscale)
                imsave(fname="%s/%i.a.png" % (out_dir, ctr), arr=this_x_processed)
                imsave(fname="%s/%i.b.png" % (out_dir, ctr), arr=pred_y_processed)
                ctr += 1

if __name__ == '__main__':
    #import pdb
    #pdb.set_trace()

    from architectures import p2p, dcgan
    model = Pix2Pix(
        gen_fn_dcgan=dcgan.default_generator,
        disc_fn_dcgan=dcgan.default_discriminator,
        gen_params_dcgan={'num_repeats':0, 'div':[2,2,4,4,8,8,8]},
        disc_params_dcgan={'num_repeats':0, 'bn':True, 'nonlinearity':linear, 'div':[8,4,4,4,2,2,2]},
        gen_fn_p2p=p2p.g_unet,
        disc_fn_p2p=p2p.discriminator,
        gen_params_p2p={'nf':64, 'act':tanh, 'num_repeats':0},
        disc_params_p2p={'nf':64, 'bn':True, 'num_repeats':0, 'act':linear, 'mul_factor':[1,2,4,8]},
        in_shp=512,
        latent_dim=1000,
        is_a_grayscale=True,
        is_b_grayscale=False
    )

    # does the p2p_gen really need to have num_repeats=1??



    
    pass
