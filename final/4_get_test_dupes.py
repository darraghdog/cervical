import os
import io
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import multiprocessing
from sklearn import cluster
import random
from scipy import misc, ndimage
from scipy.ndimage.interpolation import zoom
from scipy.ndimage import imread
from scipy.misc import imsave
from PIL import Image
#%matplotlib inline
#new_style = {'grid': False}
#plt.rc('axes', **new_style)
random.seed(100);

# Set working directory
os.chdir('/home/darragh/Dropbox/cervical/feat')
os.chdir('../data/original')

def hamdist(hash_set):
    diffs = 0
    for ch1, ch2 in zip(hash_set[0], hash_set[1]):
        if ch1 != ch2:
            diffs += 1
    return diffs

def dhash(image,hash_size = 16):
    image = image.convert('LA').resize((hash_size+1,hash_size),Image.ANTIALIAS)
    pixels = list(image.getdata())
    difference = []
    for row in xrange(hash_size):
        for col in xrange(hash_size):
            pixel_left = image.getpixel((col,row))
            pixel_right = image.getpixel((col+1,row))
            difference.append(pixel_left>pixel_right)
    decimal_value = 0
    hex_string = []
    for index, value in enumerate(difference):
        if value:
            decimal_value += 2**(index%8)
        if (index%8) == 7:
            hex_string.append(hex(decimal_value)[2:].rjust(2,'0'))
            decimal_value = 0
    return ''.join(hex_string)
    
# Lets get the train and test images and their respective gradient hashes
img_id_hash = []
parent_dir = "train"
subdir = os.listdir(parent_dir)[:3]
subdir = subdir + ['train/' + s for s in subdir] + ['test']
counter = 1
hash_size = 8
val_size = 0.2
ROWS, COLS = 64, 64

# Get the hash for each image
for direc in subdir: 
    try:
        names = os.listdir(direc)
    except:
        continue
    print counter, direc, parent_dir
    for name in names:
        if name != '.DS_Store':
            imgdata = Image.open(os.path.join(direc, name)).resize((ROWS, COLS)).convert("L")
            img_hash = dhash(imgdata, hash_size)
            img_id_hash.append([direc, name, img_hash])
            counter+=1
df = pd.DataFrame(img_id_hash,columns=['SubDirectory', 'file_name', 'image_hash'])
df = df.sort_values(['image_hash'], axis = 0).reset_index(drop=True)
df.to_csv("../../features/dupes_table_raw.csv", index = False)

# Check for dupes with test folder
dupes = []
prev_row = ['tmp', 'tmp', 'tmp']
for c, row in df.iterrows():
    curr_row = list(row)
    if hamdist([curr_row[2], prev_row[2]])<6:
        if (curr_row[0]=='test'):
            dupes.append(curr_row+prev_row)
        if (prev_row[0]=='test'):
            dupes.append(prev_row+curr_row)   
    prev_row = curr_row
    
for d in dupes[-10:]:
    img_A = imread(os.path.join(d[0], d[1]))
    img_B = imread(os.path.join(d[3], d[4]))
    plot_image = np.concatenate((img_A, img_B), axis=1)
    plt.figure(figsize=(3,3))
    plt.imshow(plot_image)
    
dupesdf = pd.DataFrame(dupes)
dupesdf = dupesdf[dupesdf[0]!=dupesdf[3]]
dupesdf.to_csv("../../features/dupes_leak6.csv", index = False, header = None)







