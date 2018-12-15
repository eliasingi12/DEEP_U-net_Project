"""
Usage:
    main.py [-a | --augment] [-t | --train] [-e | --eval] [--epochs=<epochs>]

Options:
    -h --help           Show this screen.
    -a --augment        Use STARE dataset [default: False]
    -t --train          Train model
    -e --eval           Evaluate model
    --epochs=<epochs>   N.o. epochs to train for [default: 5]
"""

import os  # misc operating system specific operations, e.g., reading directries.
import sys
import random

import cv2
import numpy as np

import matplotlib.pyplot as plt

from docopt import docopt

from unet import unet
from utils import img2bin, show_images, avg_iou, read_preproc, list_img_paths, reshape_normalize

# Some parameters and paths to data
dir_path = os.path.dirname(os.path.realpath(__file__))
PATH_DRIVE_train = 'DRIVE/training/'
PATH_DRIVE_test = 'DRIVE/test/'
PATH_DRIVE_img = 'images'
PATH_DRIVE_targets = '1st_manual_tif'
PATH_STARE_imgs = 'STARE/stare_images_tif'
PATH_STARE_masks_vk = 'STARE/labels_vk_tif'


def main(augment=False, train_model=False, eval_model=False, EPOCHS=5):
    # Read in the file paths of the images to use for the training.
    random_seed = 42
    random.seed(random_seed)

    full_paths = {}
    full_paths["training_imgs"] = os.path.join(dir_path,'..',PATH_DRIVE_train,PATH_DRIVE_img)
    full_paths["targets"] = os.path.join(dir_path,'..',PATH_DRIVE_train,PATH_DRIVE_targets)
    full_paths["testing_imgs"] = os.path.join(dir_path,'..',PATH_DRIVE_test,PATH_DRIVE_img)
    full_paths["testing_targets"] = os.path.join(dir_path,'..',PATH_DRIVE_test,PATH_DRIVE_targets)
    full_paths["STARE_imgs"] = os.path.join(dir_path, '..', PATH_STARE_imgs)
    full_paths["STARE_masks_vk"] = os.path.join(dir_path, '..', PATH_STARE_masks_vk)

    # Training set
    image_paths = list_img_paths(full_paths["training_imgs"], '.tif')
    image_paths.sort()
    target_paths = list_img_paths(full_paths["targets"], '.tif')
    target_paths.sort()

    # Testing set
    test_image_paths = list_img_paths(full_paths["testing_imgs"], '.tif')
    test_image_paths.sort()
    test_target_paths = list_img_paths(full_paths["testing_targets"], '.tif')
    test_target_paths.sort()

    if augment:
        print("Augmenting...")
        
        # Change DRIVE set to 90:10 train:test instead of 50:50 and add STARE dataset
        image_paths.extend(test_image_paths[:-4])
        test_image_paths = test_image_paths[-4:]

        target_paths.extend(test_target_paths[:-4])
        test_target_paths = test_target_paths[-4:]

        STARE_imgs = list_img_paths(full_paths["STARE_imgs"], '.tif')
        STARE_imgs.sort()
        STARE_targs = list_img_paths(full_paths["STARE_masks_vk"], '.tif')
        STARE_targs.sort()

        image_paths.extend(STARE_imgs[:-2])
        test_image_paths.extend(STARE_imgs[-2:])

        target_paths.extend(STARE_targs[:-2])
        test_target_paths.extend(STARE_targs[-2:])

        assert len(image_paths) == 54
        assert len(target_paths) == len(image_paths)
        assert len(test_image_paths) == 6
        assert len(test_target_paths) == len(test_image_paths)


    train_input = read_preproc(image_paths)
    train_target = read_preproc(target_paths)
    test_input = read_preproc(test_image_paths)
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

    model = unet(h,w,ch)
    model.summary()

    if train_model:
        model.fit(train_input, train_target, epochs=EPOCHS, batch_size=1)

    if train_model and eval_model:

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
        print("Training avg IoU: ", avg_iou(train_pred, train_target))
        print("Test avg IoU: ", avg_iou(test_pred, test_target))


if __name__ == "__main__":
    arguments = docopt(__doc__, version='0.1', help=__doc__)

    AUG = arguments['--augment']
    TRAIN = arguments['--train']
    EVAL = arguments['--eval']
    EPOCHS = arguments['--epochs']

    main(AUG, TRAIN, EVAL, int(EPOCHS))
