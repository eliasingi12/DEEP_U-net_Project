import os  # misc operating system specific operations, e.g., reading directries.
import sys
import random

import cv2
import numpy as np

import matplotlib.pyplot as plt

from unet import unet
from utils import img2bin, show_images, avg_iou, read_preproc, list_img_paths

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

args = {}
args["training_imgs"] = os.path.join(dir_path,'..',path_train,path_img)
args["targets"] = os.path.join(dir_path,'..',path_train,path_tif_targets)
args["STARE_imgs"] = os.path.join(dir_path, '..', path_STARE_imgs)
args["STARE_masks_ah"] = os.path.join(dir_path, '..', path_STARE_masks_ah)
args["STARE_masks_vk"] = os.path.join(dir_path, '..', path_STARE_masks_vk)

image_paths = []
target_paths = []

image_paths.extend(list_img_paths(args["training_imgs"]))
target_paths.exted(list_img_paths(args["targets"]))
  
image_paths.sort()
train_input = read_preproc(image_paths)
target_paths.sort()
train_target = read_preproc(target_paths)

train_input = np.array(train_input)
train_target = np.array(train_target)

#print(train_input.shape)
#print(train_target.shape)

train_input = train_input.reshape(train_input.shape[0], 512, 512, 1)
train_target = train_target.reshape(train_target.shape[0], 512, 512, 1)

train_input = train_input.astype('float32')
train_target = train_target.astype('float32')

train_input/=255
train_target/=255

h, w, ch = train_input[0].shape
#print(h)
#print(w)
#print(ch)

EPOCHS=5

model = unet(h,w,ch)
model.summary()

model.fit(train_input, train_target, epochs=EPOCHS, batch_size=1)

pred_arr = train_input[0]
pred_arr = np.expand_dims(pred_arr, axis=0)
outp = model.predict(pred_arr, batch_size=1)
outp = outp.reshape((512,512))

# show network output image
plt.imshow(outp, interpolation='nearest')
plt.show()

train_pred = model.predict(train_input)

train_target = train_target.reshape(train_target.shape[0], 512, 512)
train_pred = train_pred.reshape(train_pred.shape[0], 512, 512) 

print(avg_iou(train_pred,train_target))