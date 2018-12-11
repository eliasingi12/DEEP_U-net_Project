import os  # misc operating system specific operations, e.g., reading directries. 
import random

import cv2
import numpy as np

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt

