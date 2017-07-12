import sys
import h5py
from keras.preprocessing.image import ImageDataGenerator    
from pix2pix import Pix2Pix
from lasagne.nonlinearities import *
from lasagne.updates import *
from lasagne.utils import floatX
from util import Hdf5Iterator

def get_iterators(dataset, batch_size, is_a_grayscale, is_b_grayscale, da=True):
    dataset = h5py.File(dataset,"r")
    if da:
        imgen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, rotation_range=360, fill_mode="reflect")
    else:
        imgen = ImageDataGenerator()
    it_train = Hdf5Iterator(dataset['xt'], dataset['yt'], batch_size, imgen, is_a_grayscale=is_a_grayscale, is_b_grayscale=is_b_grayscale)
    it_val = Hdf5Iterator(dataset['xv'], dataset['yv'], batch_size, imgen, is_a_grayscale=is_a_grayscale, is_b_grayscale=is_b_grayscale)
    return it_train, it_val

if __name__ == '__main__':

    def test1_nobn(mode):
        assert mode in ["train", "interp", "gen"]
        from architectures import p2p, dcgan
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
        # models/test1_repeatnod_fixp2p_nobn/600.model.bak --> good DCGAN?? (not good p2p!!)
        # models/test1_repeatnod_fixp2p_nobn_finetunep2p_bilin/1000.model.bak --> good p2p (not good DCGAN??)
        if mode == "train":
            model.train(it_train, it_val, batch_size=bs, num_epochs=1000, out_dir="output/%s" % name, model_dir="models/%s" % name)
        elif mode == "interp":
            model.load_model("models/test1_repeatnod_fixp2p_nobn/1000.model.bak", mode='dcgan')
            model.load_model("models/test1_repeatnod_fixp2p_nobn_finetunep2p_bilin/1000.model.bak", mode='p2p')
            model.generate_interpolation_clip(100, 4, "output/%s/interp_clip_600_concat_bothdet/" % name, concat=True, deterministic=True)


    # let's use bilinear upsampling with the dcgan


    def test1_nobn_bilin_both(mode):
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
            train_mode='both'
        )
        desert_h5 = "/data/lisa/data/cbeckham/textures_v2_brown500.h5"
        bs = 4
        it_train, it_val = get_iterators(desert_h5, bs, True, False, True)
        name = "test1_nobn_bilin_both_deleteme"
        if mode == "train":
            model.train(it_train, it_val, batch_size=bs, num_epochs=1000, out_dir="output/%s" % name, model_dir="models/%s" % name)



            
            
    locals()[ sys.argv[1] ]( sys.argv[2] )
