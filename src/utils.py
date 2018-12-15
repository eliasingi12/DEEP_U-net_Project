import os
import numpy as np
import cv2

# utils.py

def read_preproc(img_paths):
    img_data = []
    for path in img_paths:
        image = cv2.imread(path)
        image = cv2.resize(image, (512, 512))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_data.append(image)
    return img_data


def list_img_paths(dir, format):
    image_paths = []
    for (dirpath, dirnames, filenames) in os.walk(dir):
        for img in filenames:
            if format in img and not img.startswith('.'):
                image_paths.append(os.path.join(dirpath, img))
    return image_paths


def show_images(imgs, grid_size=3):
    f, axarr = plt.subplots(grid_size,grid_size, figsize=(15, 15))
    for i in range(grid_size):
        for j in range(grid_size):
            axarr[i,j].imshow(imgs[i*grid_size+j])
    plt.show()


def img2bin(img):
    img_new = img.copy() # Make copy instead of changing origianl list
    rows, cols = img_new.shape
    for row in range(rows):
        for col in range(cols):
            if img_new[row][col]*255 > 215:
                img_new[row][col] = 1
            else:
                img_new[row][col] = 0
    return img_new.astype(int)

def iou(pred,target):
    intersection = pred*target
    notTrue = 1 - target
    union = target + (notTrue * pred)
    return np.sum(intersection)/np.sum(union)


def avg_iou(preds,targets):

    assert len(preds) == len(targets)
    
    bin_targets = []
    for img in targets:
        bin_targets.append(img2bin(img))

    bin_preds = []
    for img in preds:
        bin_preds.append(img2bin(img))

    pred_targets = [(bin_preds[i], bin_targets[i]) for i in range(len(bin_preds))]

    train_iou = []
    for pred, target in pred_targets:
        train_iou.append(iou(pred,target))

    return sum(train_iou)/len(preds)
  
  
def reshape_normalize(arr_to_reshape):
    arr_to_reshape = arr_to_reshape.reshape(arr_to_reshape.shape[0], 512, 512, 1)
    arr_to_reshape = arr_to_reshape.astype('float32')
    arr_to_reshape /= 255
    return arr_to_reshape
  