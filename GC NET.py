
from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Input, GlobalAveragePooling2D
from keras import backend as bk
from keras import optimizers
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.layers import Activation


import tensorflow as tf



def get_flops(model):
    run_meta = tf.RunMetadata()
    opts = tf.profiler.ProfileOptionBuilder.float_operation()
    
    # We use the Keras session graph in the call to the profiler.
    flops = tf.profiler.profile(graph=bk.get_session().graph,
                                run_meta=run_meta, cmd='op', options=opts)
                                
    return flops.total_float_ops




from keras.utils.generic_utils import get_custom_objects
def piecewise5(X):
    return bk.switch(X < -0.6, (0.01 * X ),
                     bk.switch(X < -0.2, (0.2 * X ),
                               bk.switch(X < 0.2, (1 * X ),
                                         bk.switch(X < 0.6, (1.5 * X ),
                                                   bk.switch(X < 5, (3 * X ), (3 * X ))))))

get_custom_objects().update({'piecewise5': Activation(piecewise5)})




input_shape=X_train.shape[1:]

def GC_Net(input_shape):

    
    
    input_img = Input(shape = (30, 30, 3))
    
    conv_1 = Conv2D(64, (3,3), padding='same', activation='piecewise5')(input_img)
    block1_output = GlobalAveragePooling2D()(conv_1)
    max_pool_1 = MaxPooling2D(pool_size=(2, 2),strides=(2,2), padding='same')(conv_1)
    dropout_1 = Dropout(0.25)(max_pool_1)
    
    
    
    conv_2 = Conv2D(128, (3,3), padding='same', activation='piecewise5')(dropout_1)
    block2_output = GlobalAveragePooling2D()(conv_2)
    max_pool_2 = MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding='same')(conv_2)
    dropout_2 = Dropout(0.01)(max_pool_2)
    
    
    conv_3 = Conv2D(64, (3,3), padding='same', activation='piecewise5')(dropout_2)
    block3_output = GlobalAveragePooling2D()(conv_3)
    max_pool_3 = MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding='same')(conv_3)
    dropout_3 = Dropout(0.01)(max_pool_3)
    
    
    conv_4 = Conv2D(128, (3,3), padding='same', activation='piecewise5')(dropout_3)
    block4_output = GlobalAveragePooling2D()(conv_4)
    max_pool_4 = MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding='same')(conv_4)
    dropout_4 = Dropout(0.01)(max_pool_4)
    
    
    conv_5 = Conv2D(64, (3,3), padding='same', activation='piecewise5')(dropout_4)
    block5_output = GlobalAveragePooling2D()(conv_5)
    max_pool_5 = MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding='same')(conv_5)
    dropout_5 = Dropout(0.01)(max_pool_5)
    
    
    
    conv_6 = Conv2D(128, (3,3), padding='same', activation='piecewise5')(dropout_5)
    block6_output = GlobalAveragePooling2D()(conv_6)
    
    
    
    
    
    output = keras.layers.concatenate([block1_output, block2_output, block3_output,block4_output,block5_output, block6_output ], axis = 1)
    #     output = Flatten()(output)
    output = Dense(64,activation='piecewise5')(output)
    out    = Dense(43, activation='softmax')(output)
    
    
    
    
    
    
    
    model = Model(inputs = input_img, outputs = out)
    print(model.summary())
    return model
