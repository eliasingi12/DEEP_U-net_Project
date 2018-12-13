import os  # misc operating system specific operations, e.g., reading directries.
import sys
import random

import cv2
import numpy as np

def img2bin(img):
    img_new = img.copy() # Make copy instead of changing origianl list
    rows, cols = img_new.shape
    for row in range(rows):
        for col in range(cols):
            if img_new[row][col] != 0:
                img_new[row][col] = 1
    return img_new