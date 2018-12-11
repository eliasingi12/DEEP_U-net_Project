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

# Some parameters and paths to data
dir_path = os.path.dirname(os.path.realpath(__file__))
path_train = 'DRIVE/training/'
path_test = 'DRIVE/test/'
path_img = 'images'
path_mask = 'mask'
path_targets = '1st_manual'

# Read in the file paths of the images to use for the training.
random_seed = 42
sz = 64
image_paths = []
pre_dir = "./DRIVE/"

args = {}
args["training_imgs"] = os.path.join(dir_path,path_train,path_img)
args["targets"] = os.path.join(dir_path,path_train,path_targets)

for (dirpath, dirnames, filenames) in os.walk(args["training_imgs"]):
    for file in filenames:
        if '.tif' in file and not file.startswith('.'):
              image_paths.append(os.path.join(dirpath, file))
                
random.seed(random_seed)
random.shuffle(image_paths)


def show_images(imgs, grid_size=3):
    f, axarr = plt.subplots(grid_size,grid_size, figsize=(15, 15))
    for i in range(grid_size):
        for j in range(grid_size):
            axarr[i,j].imshow(imgs[i*grid_size+j])
    plt.show()
  

input_data   = []
input_labels = []
original_imgs = []
for image_path in image_paths:
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    original_imgs.append(image)
    input_data.append(image)
    label = image_path.split(os.path.sep)[-2]
    input_labels.append(label)
    
show_images(original_imgs)

