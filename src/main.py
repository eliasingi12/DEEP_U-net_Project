import os  # misc operating system specific operations, e.g., reading directries.
import sys
import random

import cv2
import numpy as np

import matplotlib.pyplot as plt

from unet import unet
from utils import img2bin, show_images, avg_iou

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
  
image_paths.sort()
input_data = []
for image_path in image_paths:
    image = cv2.imread(image_path)
    image = cv2.resize(image, (512, 512))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    input_data.append(image)
    
target_paths.sort()
train_target = []
for target_path in target_paths:
    target = cv2.imread(target_path)
    target = cv2.resize(target, (512, 512))
    target = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    train_target.append(target)

input_data = np.array(input_data)
train_target = np.array(train_target)

#print(input_data.shape)
#print(train_target.shape)

input_data = input_data.reshape(input_data.shape[0], 512, 512, 1)
train_target = train_target.reshape(train_target.shape[0], 512, 512, 1)

input_data = input_data.astype('float32')
train_target = train_target.astype('float32')

input_data/=255
train_target/=255

h, w, ch = input_data[0].shape
#print(h)
#print(w)
#print(ch)

EPOCHS=5

model = unet(h,w,ch)
model.summary()

model.fit(input_data, train_target, epochs=EPOCHS, batch_size=1)

pred_arr = input_data[0]
pred_arr = np.expand_dims(pred_arr, axis=0)
outp = model.predict(pred_arr, batch_size=1)
outp = outp.reshape((512,512))

# show network output image
plt.imshow(outp, interpolation='nearest')
plt.show()

train_pred = model.predict(input_data)

train_target = train_target.reshape(train_target.shape[0], 512, 512)
train_pred = train_pred.reshape(train_pred.shape[0], 512, 512) 

print(avg_iou(train_pred,train_target))