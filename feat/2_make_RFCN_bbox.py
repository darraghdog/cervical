# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 21:27:44 2017

@author: darragh
"""
import csv, os, shutil
import pandas as pd
from matplotlib import pyplot as plt
import scipy
from scipy import misc, ndimage
from scipy.ndimage.interpolation import zoom
from scipy.ndimage import imread
from scipy.misc import imsave
from PIL import Image
import numpy as np
import warnings
warnings.filterwarnings('ignore')

os.chdir('/home/darragh/Dropbox/cervical/feat')
validate = True
annotations = True
validate_rfcn = False
bbimg_trn = False
bbimg_tst = False # this is the big one
TESTDIR = '../data/test'
TRAINDIR = '../data/train'
DATADIR = '../data'

def create_rect(row):
    h, w = row['h'], row['w']
    x, y = row['x0'], row['y0']
    return plt.Rectangle((x, y), w, h, color='red', fill=False, lw=2)

# Load up the train bbox file
bbox_files = ['../Type_1_bbox.tsv', '../Type_2_bboxes.tsv', '../Type_3_bbox.tsv']
bbox_ls = []
for bbox_file in bbox_files:
    with open(bbox_file, 'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            row = row[0].split(' ')
            fname = row[0].replace('\\', '/')
            coords = row[2:]
            coords = [int(i) for i in coords]
            for i in range(len(coords)/4):
                pos = i*4
                bbox_ls.append([fname] +coords[pos:(pos+4)])
bboxdf = pd.DataFrame(bbox_ls, columns=['img', 'x0', 'y0', 'w', 'h'])
bboxdf.to_csv("../boundary_boxes_from_forums.csv", index= None)

# Now check a few
if validate:
    samps = ['Type_1/787.jpg', 'Type_2/1159.jpg', 'Type_3/924.jpg']
    for samp in samps:
        samp_coords = bboxdf[bboxdf['img']==samp]
        img = imread(TRAINDIR + '/%s'%(samp))
        plt.figure(figsize=(12,8))
        plt.imshow(img)
        for c, row in samp_coords.iterrows():
            plt.gca().add_patch(create_rect(row))

if annotations:
    # #write Annotations
    import glob
    if not os.path.exists('../rfcn'):
        os.mkdir('../rfcn')
        os.mkdir('../rfcn/Annotations')
        os.mkdir('../rfcn/Annotations/train')
        for i in range(3) : 
            os.mkdir('../rfcn/Annotations/train/Type_' + str(i+1))
    c= "lesion"
    pred_ls = []
    for image in bboxdf.img.unique():
        bbox = bboxdf[bboxdf['img'] == image]
        image = 'train/' + image
        basename = image.split('.jpg')[0]
        pred_ls.append(basename)
        f = open('../rfcn/Annotations/' + basename + '.xml','w') 
        line = "<annotation>" + '\n'
        f.write(line)
        line = '\t<folder>' + c + '</folder>' + '\n'
        f.write(line)
        line = '\t<filename>' + basename + '</filename>' + '\n'
        f.write(line)
        line = '\t<source>\n\t\t<database>Source</database>\n\t</source>\n'
        f.write(line)
        im=Image.open(DATADIR + '/' + image)
        (width, height) = im.size
        line = '\t<size>\n\t\t<width>'+ str(width) + '</width>\n\t\t<height>' + \
        str(height) + '</height>\n\t\t<depth>3</depth>\n\t</size>'
        f.write(line)
        line = '\n\t<segmented>0</segmented>'
        f.write(line)
        for a in bbox.iterrows():
            a = list(a[1])
            line = '\n\t<object>'
            line += '\n\t\t<name>' + c + '</name>\n\t\t<pose>Unspecified</pose>'
            line += '\n\t\t<truncated>0</truncated>\n\t\t<difficult>0</difficult>'
            xmin = max(0.1, float(a[1]))
            line += '\n\t\t<bndbox>\n\t\t\t<xmin>' + str(xmin) + '</xmin>'
            ymin = max(0.1, float(a[2]))
            line += '\n\t\t\t<ymin>' + str(ymin) + '</ymin>'
            w = a[3]
            h = a[4]
            xmax = min(float(xmin + w), width-0.1)
            ymax = min(float(ymin + h), height-0.1)
            line += '\n\t\t\t<xmax>' + str(xmax) + '</xmax>'
            line += '\n\t\t\t<ymax>' + str(ymax) + '</ymax>'
            line += '\n\t\t</bndbox>'
            line += '\n\t</object>'     
            f.write(line)
        line = '</annotation>'
        f.write(line)
        f.close()
        
    # Create the train val datasets
    if not os.path.exists('../rfcn/ImageSets'):
        os.mkdir('../rfcn/ImageSets')
    if not os.path.exists('../rfcn/ImageSets/Main'):
        os.mkdir('../rfcn/ImageSets/Main')
               
    files = glob.glob('../data/test/*')
    for i in range(1,4):
        files += glob.glob('../data/train/Type_' + str(i) + '/*')
        files += glob.glob('../data/Type_' + str(i) + '/*')
    files = [f.replace('../data/', '').replace('.jpg', '') for f in files]
    
    trn_img = bboxdf.img.unique().tolist()
    trn_img = ['train/' + f.replace('.jpg', '') for f in trn_img]
    tst_img = list(set(files) - set(trn_img))
    
    with open('../rfcn/ImageSets/Main/trainval.txt','w') as f:
        for im in trn_img:
            f.write(im + '\n')
    
    with open('../rfcn/ImageSets/Main/train.txt','w') as f:
        for im in trn_img:
            f.write(im + '\n')
            
    with open('../rfcn/ImageSets/Main/test.txt','w') as f:
        for im in tst_img:
            f.write(im + '\n')
