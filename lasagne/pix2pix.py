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
import gzip
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
                 alpha=100, opt=adam, opt_args={'learning_rate':theano.shared(floatX(1e-3))},
                 train_mode = 'both', reconstruction='l1', sampler=np.random.rand, lsgan=False, verbose=True):
        assert train_mode in ['dcgan', 'p2p', 'both']
        self.is_a_grayscale = is_a_grayscale
        self.is_b_grayscale = is_b_grayscale
        self.latent_dim = latent_dim
        self.sampler = sampler
        self.in_shp = in_shp
        self.verbose = verbose
        self.train_mode = train_mode
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
        dcgan = {'gen':dcgan_gen, 'disc':dcgan_disc}
        dcgan['gen_out'] = get_output(dcgan_gen, Z) # G(z)
        dcgan['gen_out_det'] = get_output(dcgan_gen, Z, deterministic=True)
        dcgan['disc_out_real'] = get_output(dcgan_disc, X) # D(x)
        dcgan['disc_out_fake'] = get_output(dcgan_disc, dcgan['gen_out']) # D(G(z))
        # construct theano stuff for the p2p gen/disc
        p2p = {'gen':p2p_gen, 'disc':p2p_disc["out"]}
        p2p['disc_out_real'] = get_output(p2p_disc["out"], { p2p_disc["inputs"][0]: X, p2p_disc["inputs"][1]: Y }) # D(X,Y)
        p2p['gen_out'] = get_output(p2p_gen, X)
        p2p['gen_out_det'] = get_output(p2p_gen, X, deterministic=True)
        p2p['disc_out_fake'] = get_output(p2p_disc["out"], { p2p_disc["inputs"][0]: X, p2p_disc["inputs"][1]: p2p['gen_out'] }) # D(X, X_to_y(X))
        if lsgan:
            adv_loss = squared_error
        else:
            adv_loss = binary_crossentropy
        # dcgan loss definitions
        gen_loss_dcgan = adv_loss(dcgan['disc_out_fake'], 1.).mean()
        disc_loss_dcgan = adv_loss(dcgan['disc_out_real'], 1.).mean() + adv_loss(dcgan['disc_out_fake'], 0.).mean()
        # p2p loss definitions
        gen_loss_p2p = adv_loss(p2p['disc_out_fake'], 1.).mean()
        assert reconstruction in ['l1', 'l2']
        if reconstruction == 'l2':
            recon_loss = squared_error(p2p['gen_out'], Y).mean()
        else:
            recon_loss = T.abs_(p2p['gen_out']-Y).mean()
        #if not reconstruction_only:
        gen_total_loss_p2p = gen_loss_p2p + alpha*recon_loss
        #else:
        #    #log("GAN disabled, using only pixel-wise reconstruction loss...")
        #    gen_total_loss_p2p = recon_loss
        disc_loss_p2p = adv_loss(p2p['disc_out_real'], 1.).mean() + adv_loss(p2p['disc_out_fake'], 0.).mean()
        # dcgan params
        gen_params_dcgan = get_all_params(dcgan_gen, trainable=True)
        disc_params_dcgan = get_all_params(dcgan_disc, trainable=True)
        # pix2pix params
        gen_params_p2p = get_all_params(p2p_gen, trainable=True)
        disc_params_p2p = get_all_params(p2p_disc["out"], trainable=True)
        # --------------------
        if verbose:
            print "train_mode: %s" % train_mode
        if train_mode == 'both':
            updates = opt(gen_loss_dcgan, gen_params_dcgan, **opt_args) # update dcgan generator
            updates.update(opt(disc_loss_dcgan, disc_params_dcgan, **opt_args)) # update dcgan discriminator
            updates.update(opt(gen_total_loss_p2p, gen_params_p2p, **opt_args)) # update p2p generator
            updates.update(opt(disc_loss_p2p, disc_params_p2p, **opt_args)) # update p2p discriminator
        elif train_mode == 'dcgan':
            updates = opt(gen_loss_dcgan, gen_params_dcgan, **opt_args) # update dcgan generator
            updates.update(opt(disc_loss_dcgan, disc_params_dcgan, **opt_args)) # update dcgan discriminator
        else:
            updates = opt(gen_total_loss_p2p, gen_params_p2p, **opt_args) # update p2p generator
            updates.update(opt(disc_loss_p2p, disc_params_p2p, **opt_args)) # update p2p discriminator
        train_fn = theano.function([Z,X,Y], [gen_loss_dcgan, disc_loss_dcgan, gen_loss_p2p, recon_loss, disc_loss_p2p], updates=updates, on_unused_input='warn')
        loss_fn = theano.function([Z,X,Y], [gen_loss_dcgan, disc_loss_dcgan, gen_loss_p2p, recon_loss, disc_loss_p2p], on_unused_input='warn')
        gen_fn = theano.function([X], p2p['gen_out'])
        gen_fn_det = theano.function([X], p2p['gen_out_det'])
        z_fn = theano.function([Z], dcgan['gen_out']) # NOT DETERMINISTIC, no doesn't use bn avgs
        z_fn_det = theano.function([Z], dcgan['gen_out_det'])
        self.train_fn = train_fn
        self.loss_fn = loss_fn
        self.gen_fn = gen_fn
        self.gen_fn_det = gen_fn_det
        self.z_fn = z_fn
        self.z_fn_det = z_fn_det
        self.dcgan = dcgan
        self.p2p = p2p
        self.lr = opt_args['learning_rate']
        self.train_keys = ['dcgan_gen', 'dcgan_disc', 'p2p_gen', 'p2p_recon', 'p2p_disc']
    def save_model(self, filename):
        with gzip.open(filename, "wb") as g:
            pickle.dump({
                'dcgan': {'gen': get_all_param_values(self.dcgan['gen']), 'disc': get_all_param_values(self.dcgan['disc'])},
                'p2p': {'gen': get_all_param_values(self.p2p['gen']), 'disc': get_all_param_values(self.p2p['disc'])}
            }, g, pickle.HIGHEST_PROTOCOL )
    def load_model(self, filename, mode='both'):
        """
        filename:
        mode: what weights should we load? E.g. `both` = load
          weights for both p2p and dcgan.
        """
        assert mode in ['both', 'dcgan', 'p2p']
        with gzip.open(filename) as g:
            dd = pickle.load(g)
            if mode == 'both':
                set_all_param_values(self.dcgan['gen'], dd['dcgan']['gen'])
                set_all_param_values(self.dcgan['disc'], dd['dcgan']['disc'])                
                set_all_param_values(self.p2p['gen'], dd['p2p']['gen'])
                set_all_param_values(self.p2p['disc'], dd['p2p']['disc'])
            elif mode == 'dcgan':
                set_all_param_values(self.dcgan['gen'], dd['dcgan']['gen'])
                set_all_param_values(self.dcgan['disc'], dd['dcgan']['disc'])                
            else:
                set_all_param_values(self.p2p['gen'], dd['p2p']['gen'])
                set_all_param_values(self.p2p['disc'], dd['p2p']['disc'])
    def train(self, it_train, it_val, batch_size, num_epochs, out_dir, model_dir=None, save_every=10, resume=None, reduce_on_plateau=False, quick_run=False):
        def _loop(fn, itr):
            rec = [ [] for i in range(len(self.train_keys)) ]
            for b in range(itr.N // batch_size):
                X_batch, Y_batch = it_train.next()
                #print X_batch.shape, Y_batch.shape
                Z_batch = floatX(self.sampler(X_batch.shape[0], self.latent_dim))
                results = fn(Z_batch,X_batch,Y_batch)
                for i in range(len(results)):
                    rec[i].append(results[i])
                if quick_run:
                    break
            return tuple( [ np.mean(elem) for elem in rec ] )
        header = ["epoch"]
        for key in self.train_keys:
            header.append("train_%s" % key)
        for key in self.train_keys:
            header.append("valid_%s" % key)
        header.append("lr")
        header.append("time")
        header.append("mode")
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        if model_dir != None and not os.path.exists(model_dir):
            os.makedirs(model_dir)
        if self.verbose:
            try:
                from nolearn.lasagne.visualize import draw_to_file
                draw_to_file(get_all_layers(self.dcgan['gen']), "%s/gen_dcgan.png" % out_dir, verbose=True)
                draw_to_file(get_all_layers(self.dcgan['disc']), "%s/disc_dcgan.png" % out_dir, verbose=True)
                draw_to_file(get_all_layers(self.p2p['gen']), "%s/gen_p2p.png" % out_dir, verbose=True)
                draw_to_file(get_all_layers(self.p2p['disc']), "%s/disc_p2p.png" % out_dir, verbose=True)
            except:
                pass
        f = open("%s/results.txt" % out_dir, "wb" if resume==None else "a")
        if resume == None:
            f.write(",".join(header)+"\n"); f.flush()
            print ",".join(header)
        else:
            if self.verbose:
                print "loading weights from: %s" % resume
            self.load_model(resume)
        cb = ReduceLROnPlateau(self.lr,verbose=self.verbose)
        for e in range(num_epochs):
            out_str = []
            out_str.append(str(e+1))
            t0 = time()
            # training
            results = _loop(self.train_fn, it_train)
            for i in range(len(results)):
                #train_losses[i].append(results[i])
                out_str.append(str(results[i]))
            if reduce_on_plateau:
                cb.on_epoch_end(np.mean(recon_losses), e+1)
            # validation
            results = _loop(self.loss_fn, it_val)
            for i in range(len(results)):
                #valid_losses[i].append(results[i])
                out_str.append(str(results[i]))
            out_str.append(str(cb.learning_rate.get_value()))
            out_str.append(str(time()-t0))
            out_str.append(self.train_mode)
            out_str = ",".join(out_str)
            print out_str
            f.write("%s\n" % out_str); f.flush()
            # plot an NxN grid of [A, predict(A)]
            plot_grid("%s/out_%i.png" % (out_dir,e+1), it_val, self.gen_fn, is_a_grayscale=self.is_a_grayscale, is_b_grayscale=self.is_b_grayscale)
            # plot big pictures of predict(A) in the valid set
            self.generate_atob(it_train, 1, "%s/dump_train" % out_dir)
            self.generate_atob(it_val, 1, "%s/dump_valid" % out_dir)
            # plot A generated from G(z)
            self.generate_gz(num_examples=20, batch_size=10, out_dir="%s/dump_a" % out_dir)
            if model_dir != None and (e+1) % save_every == 0:
                self.save_model("%s/%i.model" % (model_dir, e+1))
    def generate_atob(self, itr, num_batches, out_dir, dont_predict=False):
        """

        """
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
    def generate_gz(self, num_examples, batch_size, out_dir, deterministic=True):
        """
        Generate DCGAN samples g(z).
        num_examples: number of images to generate
        batch_size: batch size
        out_dir: output folder to dump the images.
        deterministic:
        returns:
        """
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        from skimage.io import imsave
        fn = self.z_fn if not deterministic else self.z_fn_det
        z = floatX(self.sampler(num_examples, self.latent_dim))
        ctr = 0
        for b in range(num_examples // batch_size):
            out = fn(z[b*batch_size:(b+1)*batch_size])
            for i in range(out.shape[0]):
                out_processed = convert_to_rgb(out[i], is_grayscale=self.is_a_grayscale)
                imsave(fname="%s/%i.png" % (out_dir,ctr), arr=out_processed)
                ctr += 1
                
    def generate_interpolation(self, out_name, zsample1=None, zsample2=None, deterministic=True, mode='row', figsize=(10,10), cmap='gray'):
        """
        Generated an image showing the decoded interpolation between two samples
          from the prior.
        out_name: output image, which is an image grid showing the
          interpolations.
        zsample1: latent vector of size (latent_dim,). If this is `None`, then this
          will be automatically sampled from the model's pre-specified prior.
        zsample2: latent vector of size (latent_dim,). If this is `None`, then this
          will be automatically sampled from the model's pre-specified prior.
        mode: if 'row', produce a row of interpolations. If 'matrix',
          produce a matrix of interpolations.
        cmap: cmap to use with matplotlib
        returns: an output image at filename `out_name`.
        """
        import image_grid
        assert mode in ['row', 'matrix']
        fn = self.z_fn if not deterministic else self.z_fn_det
        # TODO: currently does not work with non-greyscale images
        if zsample1 == None:
            zsample1 = self.sampler(1, self.latent_dim)[0]
        if zsample2 == None:
            zsample2 = self.sampler(1, self.latent_dim)[1]
        if mode == 'row':
            grid = np.zeros( (1, 6, self.in_shp, self.in_shp, 1 if self.is_a_grayscale else 3), dtype=zsample1.dtype )
        else:
            grid = np.zeros( (5, 5, self.in_shp, self.in_shp, 1 if self.is_a_grayscale else 3), dtype=zsample1.dtype )
        ctr = 0
        if mode == 'row':
            coefs = [0.0, 0.1, 0.3, 0.6, 0.9, 1.0]
        else:
            coefs = np.linspace(0,1,25).astype(zsample1.dtype)
        if mode == 'row':
            for a in coefs:
                tmp = fn( (1-a)*zsample1[np.newaxis] + a*zsample2[np.newaxis] )
                grid[0][ctr] = convert_to_rgb(tmp[0], is_grayscale=self.is_a_grayscale)
                ctr += 1
        else:
            for y in range(5):
                for x in range(5):
                    a = coefs[ctr]
                    tmp = fn( (1-a)*zsample1[np.newaxis] + a*zsample2[np.newaxis] )
                    grid[y][x] = convert_to_rgb(tmp[0], is_grayscale=self.is_a_grayscale)
                    ctr += 1
        image_grid.write_image_grid(out_name, grid, figsize=figsize, cmap=cmap)

    def generate_interpolation_clip(self, num_samples, batch_size, out_dir, deterministic=True, min_max_norm=False, concat=False):
        """
        Generate frames corresponding to a long interpolation between
          z1, z2, ..., zn.
        num_samples: number of samples of z to interpolate between
        batch_size:
        out_dir:
        deterministic:
        min_max_norm:
        concat: if `True`, save the (a,b) pairs as single side-by-side images, otherwise
          save the a's and b's separately.
        """
        from skimage.io import imsave
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        fn = self.z_fn if not deterministic else self.z_fn_det
        fn_atob = self.gen_fn if not deterministic else self.gen_fn_det
        zs = floatX(self.sampler(num_samples, self.latent_dim))
        coefs = np.linspace(0,1,25).astype(zs.dtype)
        # generate interp tuples
        tps = [ (zs[i], zs[i+1]) for i in range(zs.shape[0]-1) ]
        all_tps = []
        for i in range(len(tps)):
            tp = tps[i]
            # generate the interps
            for a in coefs:
                all_tps.append( (1-a)*tp[0] + a*tp[1] )
        all_tps = np.asarray(all_tps, dtype=zs.dtype)
        #import pdb
        #pdb.set_trace()
        ctr = 0
        for b in range(all_tps.shape[0] // batch_size):
            z_batch = all_tps[b*batch_size:(b+1)*batch_size]
            z_out = fn(z_batch)
            p2p_out = fn_atob(z_out)
            for i in range(z_out.shape[0]):
                this_a_img = z_out[i]
                this_b_img = p2p_out[i]
                if min_max_norm:
                    this_a_img = (this_a_img - np.min(this_a_img)) / (np.max(this_a_img) - np.min(this_a_img))
                this_a_img = convert_to_rgb(this_a_img, is_grayscale=self.is_a_grayscale)
                this_b_img = convert_to_rgb(this_b_img, is_grayscale=self.is_b_grayscale)
                d = '%04d' % ctr
                if concat:
                    full_img = np.zeros( (self.in_shp, self.in_shp*2, 3), dtype=zs.dtype )
                    full_img[0:self.in_shp, 0:self.in_shp, :] = this_a_img
                    full_img[0:self.in_shp, self.in_shp::, :] = this_b_img
                    imsave(arr=full_img, fname="%s/concat_%s.png" % (out_dir, d))
                else:
                    imsave(arr=this_a_img, fname="%s/a_%s.png" % (out_dir, d))
                    imsave(arr=this_b_img, fname="%s/b_%s.png" % (out_dir, d))                
                ctr += 1

if __name__ == '__main__':
    #import pdb
    #pdb.set_trace()

    from util.data import iterate_hdf5, Hdf5Iterator
    from util.util import convert_to_rgb, compose_imgs

    def get_iterators(dataset, batch_size, is_a_grayscale, is_b_grayscale, da=True):
        dataset = h5py.File(dataset,"r")
        if da:
            imgen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, rotation_range=360, fill_mode="reflect")
        else:
            imgen = ImageDataGenerator()
        it_train = Hdf5Iterator(dataset['xt'], dataset['yt'], batch_size, imgen, is_a_grayscale=is_a_grayscale, is_b_grayscale=is_b_grayscale)
        it_val = Hdf5Iterator(dataset['xv'], dataset['yv'], batch_size, imgen, is_a_grayscale=is_a_grayscale, is_b_grayscale=is_b_grayscale)
        return it_train, it_val

    def test1_nobn(mode):
        assert mode in ["train", "interp","gen"]
        from architectures import p2p, dcgan
        # this does u(0,1) sampling, though originally we used N(0,1) i think
        model = Pix2Pix(
            gen_fn_dcgan=dcgan.default_generator,
            disc_fn_dcgan=dcgan.default_discriminator,
            gen_params_dcgan={'num_repeats':0, 'div':[2,2,4,4,8,8,8]},
            disc_params_dcgan={'num_repeats':0, 'bn':False, 'nonlinearity':linear, 'div':[8,4,4,4,2,2,2]},
            gen_fn_p2p=p2p.g_unet,
            disc_fn_p2p=p2p.discriminator,
            gen_params_p2p={'nf':64, 'act':tanh, 'num_repeats':0},
            disc_params_p2p={'nf':64, 'bn':False, 'num_repeats':0, 'act':linear, 'mul_factor':[1,2,4,8]},
            in_shp=512,
            latent_dim=1000,
            is_a_grayscale=True,
            is_b_grayscale=False,
            lsgan=True,
            opt=rmsprop,
            opt_args={'learning_rate':theano.shared(floatX(1e-4))}
        )
        desert_h5 = "/data/lisa/data/cbeckham/textures_v2_brown500.h5"
        bs = 4
        it_train, it_val = get_iterators(desert_h5, bs, True, False, True)
        name = "test1_repeatnod_fixp2p_nobn"
        if mode == "train":
            model.train(it_train, it_val, batch_size=bs, num_epochs=1000, out_dir="output/%s" % name, model_dir="models/%s" % name)
        elif mode == "interp":
            model.load_model("models/%s/600.model.bak" % name)
            zs = model.sampler(2, model.latent_dim)
            z1, z2 = floatX(zs[0]), floatX(zs[1])
            model.generate_interpolation(z1, z2, "/tmp/test.png", mode='matrix')
        elif mode == "gen":
            model.load_model("models/%s/600.model.bak" % name)
            model.generate_gz(100, 10, "deleteme")

            
    def test1_nobn_finetunep2p_bilin(mode):
        assert mode in ["train", "interp","gen"]
        from architectures import p2p, dcgan
        # change the p2p discriminator
        model = Pix2Pix(
            gen_fn_dcgan=dcgan.default_generator,
            disc_fn_dcgan=dcgan.default_discriminator,
            gen_params_dcgan={'num_repeats':0, 'div':[2,2,4,4,8,8,8]},
            disc_params_dcgan={'num_repeats':0, 'bn':False, 'nonlinearity':linear, 'div':[8,4,4,4,2,2,2]},
            gen_fn_p2p=p2p.g_unet,
            disc_fn_p2p=p2p.discriminator,
            gen_params_p2p={'nf':64, 'act':tanh, 'num_repeats':0, 'bilinear_upsample':True},
            disc_params_p2p={'nf':64, 'bn':False, 'num_repeats':0, 'act':linear, 'mul_factor':[1,2,4,8]},
            in_shp=512,
            latent_dim=1000,
            is_a_grayscale=True,
            is_b_grayscale=False,
            lsgan=True,
            opt=rmsprop,
            opt_args={'learning_rate':theano.shared(floatX(1e-4))},
            train_mode='p2p'
        )
        model.load_model("models/test1_repeatnod_fixp2p_nobn/1000.model.bak", mode='dcgan') # only load the dcgan
        desert_h5 = "/data/lisa/data/cbeckham/textures_v2_brown500.h5"
        bs = 4
        it_train, it_val = get_iterators(desert_h5, bs, True, False, True)
        name = "test1_repeatnod_fixp2p_nobn_finetunep2p_bilin"
        if mode == "train":
            model.train(it_train, it_val, batch_size=bs, num_epochs=1000, out_dir="output/%s" % name, model_dir="models/%s" % name)
        elif mode == "interp":
            model.load_model("models/test1_repeatnod_fixp2p_nobn/1000.model.bak", mode='dcgan')
            model.load_model("models/test1_repeatnod_fixp2p_nobn_finetunep2p_bilin/1000.model.bak", mode='p2p')
            # models/test1_repeatnod_fixp2p_nobn/600.model.bak --> good DCGAN?? (not good p2p!!)
            # models/test1_repeatnod_fixp2p_nobn_finetunep2p_bilin/1000.model.bak --> good p2p (not good DCGAN??)
            model.generate_interpolation_clip(100, 4, "output/%s/interp_clip_600_concat_bothdet/" % name, concat=True, deterministic=True)
    
    locals()[ sys.argv[1] ]( sys.argv[2] )
