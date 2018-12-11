import os  # misc operating system specific operations, e.g., reading directries. 
import random

import cv2
import numpy as np

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

from keras.layers import Input
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt

# unet((584,565,1))

def unet(input_shape):

    input_shape = Input(input_shape)

    c1 = keras.layers.Conv2D(input_shape,n_filters=1,kernel_size=3,padding='same',
                            kernel_initializer='random_uniform')
    
    return model