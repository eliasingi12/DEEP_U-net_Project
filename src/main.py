import os  # misc operating system specific operations, e.g., reading directries.
import sys
import random

import cv2
import numpy as np

import matplotlib.pyplot as plt

from unet import unet

#resize inputs & targets
import skimage
from skimage.transform import resize

# Some parameters and paths to data
dir_path = os.path.dirname(os.path.realpath(__file__))
path_train = 'DRIVE/training/'
path_test = 'DRIVE/test/'
path_img = 'images'
path_mask = 'mask'
path_targets = '1st_manual'
path_tif_targets = '1st_manual_tif'

# Read in the file paths of the images to use for the training.
random_seed = 42
sz = 64
image_paths = []
target_paths = []

args = {}
args["training_imgs"] = os.path.join(dir_path,'..',path_train,path_img)
args["targets"] = os.path.join(dir_path,'..',path_train,path_tif_targets)

for (dirpath, dirnames, filenames) in os.walk(args["training_imgs"]):
    for file in filenames:
        if '.tif' in file and not file.startswith('.'):
              image_paths.append(os.path.join(dirpath, file))

for (dirpath, dirnames, filenames) in os.walk(args["targets"]):
    for file in filenames:
        if '.tiff' in file and not file.startswith('.'):
              target_paths.append(os.path.join(dirpath, file))
                
random.seed(random_seed)

def show_images(imgs, grid_size=3):
    f, axarr = plt.subplots(grid_size,grid_size, figsize=(15, 15))
    for i in range(grid_size):
        for j in range(grid_size):
            axarr[i,j].imshow(imgs[i*grid_size+j])
    plt.show()
  
image_paths.sort()
input_data = []
for image_path in image_paths:
    image = cv2.imread(image_path)
    image = resize(image, (560, 560), anti_aliasing=True)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    input_data.append(image)
    
target_paths.sort()
target_data = []
for target_path in target_paths:
    target = cv2.imread(target_path)
    target = resize(target, (560, 560), anti_aliasing=True)
    #target = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    #print(target)
    target_data.append(target)
    #print(target_data)

h,w,ch = target_data[0].shape

#print(target_data[0].shape)
#print(input_data[0].shape)
#show_images(target_data)

EPOCHS=5

model = unet(h,w,ch)
model.summary()


input_data = np.array(input_data)
target_data = np.array(target_data)
print(target_data.shape)
print(input_data.shape)
#input_data = np.reshape(input_data, (20,584,565,3))

model.fit(input_data, target_data, epochs=EPOCHS, batch_size=1)