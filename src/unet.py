import os  # misc operating system specific operations, e.g., reading directries. 
import random

import cv2
import numpy as np

from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Cropping2D
from keras.models import Sequential, Model
from keras.layers.core import Dense
from keras.optimizers import SGD
from sklearn.metrics import classification_report


def unet(height,width,n_ch):
    inputs = Input((height,width,n_ch))

    # First set of layers
    conv1 = Conv2D(64, (3,3), padding='same', kernel_initializer='random_uniform', activation='relu')(inputs)
    conv1 = Conv2D(64, (3,3), padding='same', kernel_initializer='random_uniform', activation='relu')(conv1)
    pool1 = MaxPooling2D((2,2))(conv1)

    # Second set of layers
    conv2 = Conv2D(128, (3,3), padding='same', kernel_initializer='random_uniform', activation='relu')(pool1)
    conv2 = Conv2D(128, (3,3), padding='same', kernel_initializer='random_uniform', activation='relu')(conv2)
    pool2 = MaxPooling2D((2,2))(conv2)

    # Third set of layers
    conv3 = Conv2D(256, (3,3), padding='same', kernel_initializer='random_uniform', activation='relu')(pool2)
    conv3 = Conv2D(256, (3,3), padding='same', kernel_initializer='random_uniform', activation='relu')(conv3)
    pool3 = MaxPooling2D((2,2))(conv3)

    # Fourth set of layers
    conv4 = Conv2D(512, (3,3), padding='same', kernel_initializer='random_uniform', activation='relu')(pool3)
    conv4 = Conv2D(512, (3,3), padding='same', kernel_initializer='random_uniform', activation='relu')(conv4)
    pool4 = MaxPooling2D((2,2))(conv4)

    # Fifth set of layers
    conv5 = Conv2D(1024, (3,3), padding='same', kernel_initializer='random_uniform', activation='relu')(pool4)
    conv5 = Conv2D(1024, (3,3), padding='same', kernel_initializer='random_uniform', activation='relu')(conv5)

    # First up layers
    upsamp1 = UpSampling2D((2,2))(conv5)
    #crop1 = Cropping2D(cropping=((0,0),(0,0)))(conv4)
    concat1 = concatenate([upsamp1,conv4])

    conv6 = Conv2D(512, (3,3), padding='same', kernel_initializer='random_uniform', activation='relu')(concat1)
    conv6 = Conv2D(512, (3,3), padding='same', kernel_initializer='random_uniform', activation='relu')(conv6)

    # Second up layers
    upsamp2 = UpSampling2D((2,2))(conv6)
    #crop2 = Cropping2D(cropping=((0,0),(0,0)))(conv3)
    concat2 = concatenate([upsamp2,conv3])

    conv7 = Conv2D(256, (3,3), padding='same', kernel_initializer='random_uniform', activation='relu')(concat2)
    conv7 = Conv2D(256, (3,3), padding='same', kernel_initializer='random_uniform', activation='relu')(conv7)

    # Third up layers
    upsamp3 = UpSampling2D((2,2))(conv7)
    #crop3 = Cropping2D(cropping=((0,0),(0,0)))(conv2)
    concat3 = concatenate([upsamp3,conv2])

    conv8 = Conv2D(128, (3,3), padding='same', kernel_initializer='random_uniform', activation='relu')(concat3)
    conv8 = Conv2D(128, (3,3), padding='same', kernel_initializer='random_uniform', activation='relu')(conv8)

    # Fourth up layers
    upsamp4 = UpSampling2D((2,2))(conv8)
    #crop4 = Cropping2D(cropping=((0,0),(0,0)))(conv1)
    concat4 = concatenate([upsamp4,conv1])

    conv9 = Conv2D(64, (3,3), padding='same', kernel_initializer='random_uniform', activation='relu')(concat4)
    conv9 = Conv2D(64, (3,3), padding='same', kernel_initializer='random_uniform', activation='relu')(conv9)

    # Output layer
    outconv = Conv2D(3, (1,1), kernel_initializer='random_uniform', padding='same', activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=outconv)

    #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.3, nesterov=False)
    model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

    return model