from eugene_fcnlib import resunet
from eugene_fcnlib.blocks import (basic_block_mp,
                        basic_block,
                        bottleneck)
from collections import OrderedDict
from keras.layers import Conv2D
from keras.models import Model
from keras.activations import relu

model_kwargs = OrderedDict((
    ('input_shape', (1, 512, 512)),
    ('num_classes', None),
    ('input_num_filters', 8),
    ('main_block_depth', [1, 1, 1, 1, 1, 1]),
    ('num_main_blocks', 5),
    ('num_init_blocks', 1),
    ('weight_decay', 0.0001), 
    ('dropout', 0.1),
    ('short_skip', True),
    ('long_skip', True),
    ('long_skip_merge_mode', 'sum'),
    ('use_skip_blocks', False),
    ('relative_num_across_filters', 1),
    ('mainblock', bottleneck),
    ('initblock', basic_block_mp)
    ))

def leaky_relu(x):
    return relu(x, alpha=0.2)

def change_relu_to_leaky_relu(model):
    for layer in model.layers:
        if hasattr(layer, 'activation'):
            if layer.activation == relu:
                layer.activation = leaky_relu

# https://arxiv.org/pdf/1512.03385.pdf
# fig 5 for 'mainblock' illustration

model = resunet.assemble_model(**model_kwargs)
x = model.layers[-1].output
x = Conv2D(filters=3, kernel_size=3, activation='tanh', padding='same')(x)
model1 = Model(input=model.layers[0].output,output=x)
change_relu_to_leaky_relu(model1)

print "model num params:", model1.count_params()

from keras.utils import plot_model
plot_model(model1, show_shapes=True, to_file="test_eugene_model.png")
