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
    conv1 = Conv2D(64, (3,3), padding='same', kernel_initializer='random_uniform', activation='relu', data_format='channels_last')(inputs)
    conv1 = Conv2D(64, (3,3), padding='same', kernel_initializer='random_uniform', activation='relu', data_format='channels_last')(conv1)
    pool1 = MaxPooling2D((2,2))(conv1)

    # Second set of layers
    conv2 = Conv2D(128, (3,3), padding='same', kernel_initializer='random_uniform', activation='relu', data_format='channels_last')(pool1)
    conv2 = Conv2D(128, (3,3), padding='same', kernel_initializer='random_uniform', activation='relu', data_format='channels_last')(conv2)
    pool2 = MaxPooling2D((2,2))(conv2)

    # Third set of layers
    conv3 = Conv2D(256, (3,3), padding='same', kernel_initializer='random_uniform', activation='relu', data_format='channels_last')(pool2)
    conv3 = Conv2D(256, (3,3), padding='same', kernel_initializer='random_uniform', activation='relu', data_format='channels_last')(conv3)
    pool3 = MaxPooling2D((2,2))(conv3)

    # Fourth set of layers
    conv4 = Conv2D(512, (3,3), padding='same', kernel_initializer='random_uniform', activation='relu', data_format='channels_last')(pool3)
    conv4 = Conv2D(512, (3,3), padding='same', kernel_initializer='random_uniform', activation='relu', data_format='channels_last')(conv4)
    pool4 = MaxPooling2D((2,2))(conv4)

    # Fifth set of layers
    conv5 = Conv2D(1024, (3,3), padding='same', kernel_initializer='random_uniform', activation='relu', data_format='channels_last')(pool4)
    conv5 = Conv2D(1024, (3,3), padding='same', kernel_initializer='random_uniform', activation='relu', data_format='channels_last')(conv5)

    # First up layers
    upsamp1 = UpSampling2D((2,2))(conv5)
    crop1 = Cropping2D(cropping=((1,0),(0,0)))(conv4)
    concat1 = concatenate([upsamp1,crop1])

    conv6 = Conv2D(512, (3,3), padding='same', kernel_initializer='random_uniform', activation='relu', data_format='channels_last')(concat1)
    conv6 = Conv2D(512, (3,3), padding='same', kernel_initializer='random_uniform', activation='relu', data_format='channels_last')(conv6)

    # Second up layers
    upsamp2 = UpSampling2D((2,2))(conv6)
    crop2 = Cropping2D(cropping=((1,1),(1,0)))(conv3)
    concat2 = concatenate([upsamp2,crop2])

    conv7 = Conv2D(256, (3,3), padding='same', kernel_initializer='random_uniform', activation='relu', data_format='channels_last')(concat2)
    conv7 = Conv2D(256, (3,3), padding='same', kernel_initializer='random_uniform', activation='relu', data_format='channels_last')(conv7)

    # Third up layers
    upsamp3 = UpSampling2D((2,2))(conv7)
    crop3 = Cropping2D(cropping=((2,2),(1,1)))(conv2)
    concat3 = concatenate([upsamp3,crop3])

    conv8 = Conv2D(128, (3,3), padding='same', kernel_initializer='random_uniform', activation='relu', data_format='channels_last')(concat3)
    conv8 = Conv2D(128, (3,3), padding='same', kernel_initializer='random_uniform', activation='relu', data_format='channels_last')(conv8)

    # Fourth up layers
    upsamp4 = UpSampling2D((2,2))(conv8)
    crop4 = Cropping2D(cropping=((4,4),(2,3)))(conv1)
    concat4 = concatenate([upsamp4,crop4])

    conv9 = Conv2D(64, (3,3), padding='same', kernel_initializer='random_uniform', activation='relu', data_format='channels_last')(concat4)
    conv9 = Conv2D(64, (3,3), padding='same', kernel_initializer='random_uniform', activation='relu', data_format='channels_last')(conv9)

    # Output layer
    outconv = Conv2D(2, (1,1), kernel_initializer='random_uniform', activation='sigmoid', data_format='channels_last')(conv9)

    model = Model(inputs=inputs, outputs=outconv)

    #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.3, nesterov=False)
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

    return model