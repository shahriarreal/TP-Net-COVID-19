from __future__ import print_function
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Input, GlobalAveragePooling2D, AveragePooling2D
from keras import backend as bk
from keras import optimizers
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.layers import Activation
from keras import initializers, regularizers
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

def custom_network(height, width, channels, classes, pre_trained=''):
    weight_decay = 0.0001
    input_img = Input(shape=(height, width, channels))
    if pre_trained != '':
        base_model = load_model(pre_trained)
        num_layers = len(base_model.layers)
        base_output = base_model.get_layer(index=num_layers-2).output
        out = Dense(classes, activation='softmax')(base_output)
        model = Model(inputs=base_model.input, outputs=out)
    else:
        conv_1a = Conv2D(64, (3, 3), padding='same', activation='relu')(input_img)
        conv_1b = Conv2D(64, (3, 3), padding='same', activation='relu')(conv_1a)    
        max_pool_1 = MaxPooling2D((2,2), strides=(2, 2), padding='same')(conv_1b)

        conv_2a = Conv2D(128, (3, 3), padding='same', activation='relu')(max_pool_1)
        conv_2b = Conv2D(128, (3, 3), padding='same', activation='relu')(conv_2a)    
        max_pool_2 = MaxPooling2D((2,2), strides=(2, 2), padding='same')(conv_2b)

        conv_3a = Conv2D(256, (3, 3), padding='same', activation='relu')(max_pool_2)
        conv_3b = Conv2D(256, (3, 3), padding='same', activation='relu')(conv_3a)
        conv_3c = Conv2D(256, (3, 3), padding='same', activation='relu')(conv_3b)    
        max_pool_3 = MaxPooling2D((2,2), strides=(2, 2), padding='same')(conv_3c)

        conv_4a = Conv2D(512, (3, 3), padding='same', activation='relu')(max_pool_3)
        conv_4b = Conv2D(512, (3, 3), padding='same', activation='relu')(conv_4a)
        conv_4c = Conv2D(512, (3, 3), padding='same', activation='relu')(conv_4b)    
        max_pool_4 = MaxPooling2D((2,2), strides=(2, 2), padding='same')(conv_4c)

        conv_5a = Conv2D(512, (3, 3), padding='same', activation='relu')(max_pool_4)
        conv_5b = Conv2D(512, (3, 3), padding='same', activation='relu')(conv_5a)
        conv_5c = Conv2D(512, (3, 3), padding='same', activation='relu')(conv_5b)    
        max_pool_5 = MaxPooling2D((2,2), strides=(2, 2), padding='same')(conv_5c)

        output = Flatten()(max_pool_5)
        
        output = Dense(4096, activation='relu')(output)
        output = Dense(4096, activation='relu')(output)
        output = Dense(classes, activation='softmax')

        model = Model(inputs=input_img, outputs=out)
    
    print(model.summary())
    return model


