import numpy as np

def img2bin(img):
    img_new = img.copy() # Make copy instead of changing origianl list
    rows, cols = img_new.shape
    for row in range(rows):
        for col in range(cols):
            if img_new[row][col]*255 > 100:
                img_new[row][col] = 255
            else:
                img_new[row][col] = 0
    return img_new


def show_images(imgs, grid_size=3):
    f, axarr = plt.subplots(grid_size,grid_size, figsize=(15, 15))
    for i in range(grid_size):
        for j in range(grid_size):
            axarr[i,j].imshow(imgs[i*grid_size+j])
    plt.show()


def iou(pred,target):
    intersection = pred*target
    notTrue = 1 - target
    union = target + (notTrue * pred)
    return sum(intersection)/sum(union)


def avg_iou(preds,targtes):

    assert len(preds) == len(targets)
    
    targets = []
    for img in train_target:
        targets.append(img2bin(img))

    preds = []
    for img in train_pred:
        preds.append(img2bin(img))

    pred_targets = [(preds[i], targets[i]) for i in range(len(preds))]

    train_iou = []
    for pred, target in pred_targets:
        train_iou.append(iou(pred,target))

    return sum(train_iou)/len(preds)