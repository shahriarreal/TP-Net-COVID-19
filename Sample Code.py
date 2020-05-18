# Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
from PIL import Image
import os,sys


# Reading the input images and putting them into a numpy array
data=[]
labels=[]

height = 30
width = 30
channels = 3
classes = 43
n_inputs = height * width*channels

for i in range(classes) :
    path = "/home/sreal/gtsrb-german-traffic-sign/Train/{0}/".format(i)
    print(path)
    Class=os.listdir(path)
    for a in Class:
        try:
            image=cv2.imread(path+a)
            image_from_array = Image.fromarray(image, 'RGB')
            size_image = image_from_array.resize((height, width))
            data.append(np.array(size_image))
            labels.append(i)
        except AttributeError:
            print(" ")

Cells=np.array(data)
labels=np.array(labels)

#Randomize the order of the input images
s=np.arange(Cells.shape[0])
np.random.seed(43)
np.random.shuffle(s)
Cells=Cells[s]
labels=labels[s]



#Spliting the images into train and validation sets
(X_train,X_val)=Cells[(int)(0.2*len(labels)):],Cells[:(int)(0.2*len(labels))]
X_train = X_train.astype('float32')/255
X_val = X_val.astype('float32')/255
(y_train,y_val)=labels[(int)(0.2*len(labels)):],labels[:(int)(0.2*len(labels))]

#Using one hote encoding for the train and validation labels
from keras.utils import to_categorical
y_train = to_categorical(y_train, 43)
y_val = to_categorical(y_val, 43)


from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Input, GlobalAveragePooling2D, AveragePooling2D
from keras import backend as bk
from keras import optimizers
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.layers import Activation
from keras import initializers

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

def custom_network(input_shape):

    input_img = Input(shape = (30, 30, 3))
    tower_1 = Conv2D(16, (1,1), padding='same', activation='elu', bias_initializer=initializers.Constant(.1))(input_img)
    tower_x = Conv2D(32, (3,3), padding='same', activation='elu', bias_initializer=initializers.Constant(.1))(tower_1)
    block1_output = GlobalAveragePooling2D()(tower_1)
    tower_y = MaxPooling2D(pool_size=(2, 2), padding='same')(tower_x)
    tower_y = Dropout(0.1)(tower_y)
    tower_z = Conv2D(32, (1,1), padding='same', activation='elu', bias_initializer=initializers.Constant(.1))(tower_y)
    tower_a = Conv2D(32, (3,3), padding='same', activation='elu', bias_initializer=initializers.Constant(.1))(tower_z)
    tower_a = MaxPooling2D(pool_size=(2, 2), padding='same')(tower_a)
    tower_a = Dropout(0.1)(tower_a)
    
    
    tower_2 = AveragePooling2D(pool_size=(4, 4), padding='same')(tower_x)
    tower_2 = Dropout(0.1)(tower_2)
    
    tower_3 = AveragePooling2D(pool_size=(2, 2), padding='same')(tower_z)
    tower_3 = Dropout(0.1)(tower_3)
    
    
    
    
    
    
    
    
    
    output = keras.layers.concatenate([tower_a, tower_2, tower_3], axis = 1)
    output = Flatten()(output)
    out1 = keras.layers.concatenate([output,block1_output],axis=1)
    
    out    = Dense(43, activation='softmax')(out1)
    
    
    
    
    
    
    
    model = Model(inputs = input_img, outputs = out)
    print(model.summary())
    return model





sgd=optimizers.SGD(lr=0.001, momentum=0.9, nesterov=False)
model = custom_network(input_shape)



model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])

# #Compilation of the model
# model.compile(
#     loss='categorical_crossentropy',
#     optimizer='adam',
#     metrics=['accuracy']
#
#using ten epochs for the training and saving the accuracy for each epoch
epochs = 20
hist1 = model.fit(X_train, y_train, batch_size=32, epochs=epochs,
                  validation_data=(X_val, y_val))

score = model.evaluate(X_val, y_val, verbose=0)
score2=model.evaluate(X_train,y_train, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
print('Training loss:', score2[0])
print('Training accuracy:', score2[1])
