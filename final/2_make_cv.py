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
#%matplotlib inline
#new_style = {'grid': False}
#plt.rc('axes', **new_style)
random.seed(100);

# Set working directory
os.chdir('/home/darragh/Dropbox/cervical/feat')
os.chdir('../data')

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
subdir = subdir + ['train/' + s for s in subdir]
counter = 1
hash_size = 4
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
df = df.sort(['image_hash'], axis = 0).reset_index(drop=True)
df.to_csv("../features/hash_table_raw.csv", index = False)

# Now lets just do a sort and cut the first 25% to get our validation set
val_img = []
val_len = int(df.shape[0]*val_size)
for c, row in df[:val_len].reset_index(drop=True).iterrows():
    val_img.append(row[0] + '/' + row[1])

fo = open('../val_images.csv','w')
for i in val_img:
    fo.write('%s\n'%(i))
fo.close()	

# Now we split into CV 5-fold
val_split = []
for i in range(5):
    val_split += [i]*((df.shape[0]/5)+1)
df['fold'] = val_split[:df.shape[0]]
df.to_csv("../cv_split.csv", index = False)
