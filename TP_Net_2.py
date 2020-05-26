from __future__ import print_function
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Input, GlobalAveragePooling2D, AveragePooling2D
from keras import backend as bk
from keras import optimizers
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.layers import Activation
from keras import initializers
import tensorflow as tf
from keras.utils.generic_utils import get_custom_objects
from keras.models import load_model


def get_flops(model):
    run_meta = tf.RunMetadata()
    opts = tf.profiler.ProfileOptionBuilder.float_operation()

    # We use the Keras session graph in the call to the profiler.
    flops = tf.profiler.profile(graph=bk.get_session().graph,
                                run_meta=run_meta, cmd='op', options=opts)

    return flops.total_float_ops


def piecewise5(X):
    return bk.switch(X < -0.6, (0.01 * X ),
                     bk.switch(X < -0.2, (0.2 * X ),
                               bk.switch(X < 0.2, (1 * X ),
                                         bk.switch(X < 0.6, (1.5 * X ),
                                                   bk.switch(X < 5, (3 * X ), (3 * X )))))) 


def custom_network(height, width, classes, pre_trained=''):
    
    input_img = Input(shape=(height, width, 3))
    if pre_trained != '':
        base_model = load_model(pre_trained)
        num_layers = len(base_model.layers)
        base_output = base_model.get_layer(index=num_layers-2).output
        out = Dense(classes, activation='softmax')(base_output)
        model = Model(inputs=base_model.input, outputs=out)
    else:
        tower_1 = Conv2D(16, (1, 1), padding='same', activation='elu', bias_initializer=initializers.Constant(.1))(input_img)
        tower_x = Conv2D(32, (3, 3), padding='same', activation='elu', bias_initializer=initializers.Constant(.1))(tower_1)
        block1_output = GlobalAveragePooling2D()(tower_1)
        tower_y = MaxPooling2D(pool_size=(2, 2), padding='same')(tower_x)
        tower_y = Dropout(0.1)(tower_y)
        tower_z = Conv2D(32, (1, 1), padding='same', activation='elu', bias_initializer=initializers.Constant(.1))(tower_y)
        tower_a = Conv2D(32, (3, 3), padding='same', activation='elu', bias_initializer=initializers.Constant(.1))(tower_z)
        tower_a = MaxPooling2D(pool_size=(2, 2), padding='same')(tower_a)
        tower_a = Dropout(0.1)(tower_a)

        tower_2 = AveragePooling2D(pool_size=(4, 4), padding='same')(tower_x)
        tower_2 = Dropout(0.1)(tower_2)

        tower_3 = AveragePooling2D(pool_size=(2, 2), padding='same')(tower_z)
        tower_3 = Dropout(0.1)(tower_3)

        output = keras.layers.concatenate([tower_a, tower_2, tower_3], axis=1)
        output = Flatten()(output)
        out1 = keras.layers.concatenate([output, block1_output], axis=1)

        out = Dense(classes, activation='softmax')(out1)

        model = Model(inputs=input_img, outputs=out)

    print(model.summary())
    return model

