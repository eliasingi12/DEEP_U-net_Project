import os  # misc operating system specific operations, e.g., reading directries.
import sys
import random

import cv2
import numpy as np

import matplotlib.pyplot as plt

from unet import unet
from utils import img2bin, show_images, avg_iou, read_preproc, list_img_paths, reshape_normalize

# Some parameters and paths to data
dir_path = os.path.dirname(os.path.realpath(__file__))
path_train = 'DRIVE/training/'
path_test = 'DRIVE/test/'
path_img = 'images'
path_mask = 'mask'
path_targets = '1st_manual'
path_tif_targets = '1st_manual_tif'
path_STARE_imgs = 'STARE/stare_images'
path_STARE_masks_vk = 'STARE/labels_vk'
path_STARE_masks_ah = 'STARE/labels_ah'

# Read in the file paths of the images to use for the training.
random_seed = 42
random.seed(random_seed)
sz = 64

args = {}
args["training_imgs"] = os.path.join(dir_path,'..',path_train,path_img)
args["targets"] = os.path.join(dir_path,'..',path_train,path_tif_targets)
args["testing_imgs"] = os.path.join(dir_path,'..',path_test,path_img)
args["testing_targets"] = os.path.join(dir_path,'..',path_test,path_tif_targets)
args["STARE_imgs"] = os.path.join(dir_path, '..', path_STARE_imgs)
args["STARE_masks_ah"] = os.path.join(dir_path, '..', path_STARE_masks_ah)
args["STARE_masks_vk"] = os.path.join(dir_path, '..', path_STARE_masks_vk)

image_paths = []
target_paths = []
test_image_paths = []
test_target_paths = []

image_paths.extend(list_img_paths(args["training_imgs"], '.tif'))
target_paths.extend(list_img_paths(args["targets"], '.tif'))
test_image_paths.extend(list_img_paths(args["testing_imgs"], '.tif'))
test_target_paths.extend(list_img_paths(args["testing_targets"], '.tif'))

image_paths.sort()
train_input = read_preproc(image_paths)
target_paths.sort()
train_target = read_preproc(target_paths)
test_image_paths.sort()
test_input = read_preproc(test_image_paths)
test_target_paths.sort()
test_target = read_preproc(test_target_paths)

train_input = np.array(train_input)
train_target = np.array(train_target)
test_input = np.array(test_input)
test_target = np.array(test_target)

train_input = reshape_normalize(train_input)
train_target = reshape_normalize(train_target)
test_input = reshape_normalize(test_input)
test_target = reshape_normalize(test_target)

h, w, ch = train_input[0].shape
EPOCHS=1

model = unet(h,w,ch)
model.summary()

model.fit(train_input, train_target, epochs=EPOCHS, batch_size=1)

# training prediction
train_pred = model.predict(train_input)

# testing prediction
test_pred = model.predict(test_input)

### Display Images ###
train_target = train_target.reshape(train_target.shape[0], 512, 512)
test_target = test_target.reshape(test_target.shape[0], 512, 512)
train_pred = train_pred.reshape(train_pred.shape[0], 512, 512) 
test_pred = test_pred.reshape(test_pred.shape[0], 512, 512) 

plt.imshow(train_pred[0], interpolation='nearest')
plt.show()

plt.imshow(train_target[0], interpolation='nearest')
plt.show()

plt.imshow(test_pred[0], interpolation='nearest')
plt.show()

plt.imshow(test_target[0], interpolation='nearest')
plt.show()
### ###

# check iou of test and train data
train_sum = 0
test_sum = 0
for i in range(len(train_pred)):
    print("image: " + str(i) + "...")
    train_iou = iou(img2bin(train_pred[i]), img2bin(train_target[i]))
    test_iou = iou(img2bin(test_pred[i]), img2bin(test_target[i]))               
    train_sum += train_iou
    test_sum += test_iou
    
print("train_avg: ")
print(train_sum/len(train_pred))

print("test_avg: ")
print(test_sum/len(train_pred))