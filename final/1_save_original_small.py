# -*- coding: utf-8 -*-
"""
Created on Tue May 16 19:34:34 2017

@author: darragh
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import math
from sklearn import mixture
from sklearn.utils import shuffle
from skimage import measure
from glob import glob
import os
from PIL import Image


os.chdir('/home/darragh/Dropbox/cervical/feat')
TRAIN_DATA = "../data/train"
TEST_DATA = "../data/test"
ADDITIONAL_DATA = "../data"
train_files = glob(os.path.join(TRAIN_DATA, "*", "*.jpg"))
test_files = glob(os.path.join(TEST_DATA, "*.jpg"))
additional_files = glob(os.path.join(ADDITIONAL_DATA, 'Type_*', "*.jpg"))
ROWS = 224
COLS = 224

# Create directories
if not os.path.exists('../data/original'):
    os.mkdir('../data/original')
    os.mkdir('../data/original/test')
    os.mkdir('../data/original/train')
    for i in range(3) : 
        os.mkdir('../data/original/Type_' + str(i+1))
        os.mkdir('../data/original/train/Type_' + str(i+1))

# Write images
i = 0 
all_paths = train_files + test_files + additional_files
for f in sorted(all_paths):
    if i % 50 == 0 : print('Processing image {} out of {}'.format(i, len(all_paths)))
    fpath = f.replace('../data', '../data/original')
    img = Image.open(f)
    try:
        img = img.convert('RGB')
    except:
        print('Failed to read {}'.format(fpath))
        continue
    img = img.resize((COLS * 2, ROWS * 2))
    img.save(fpath)
    i += 1