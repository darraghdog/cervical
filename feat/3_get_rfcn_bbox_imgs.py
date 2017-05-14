# -*- coding: utf-8 -*-
"""
Created on Sat May 13 14:22:10 2017

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
SAVE_BBOX = True
TESTDIR = '../data/test'
TRAINDIR = '../data/train'
DATACROPDIR = '../data/rfcn_crop'
TESTCROPDIR = '../data/rfcn_crop/test'
TRAINCROPDIR = '../data/rfcn_train'
DATADIR = '../data'
ROWS, COLS = 224, 224
MIN_DIMENSION = 224*7

def create_rect(row):
    w, h = row['x1'] - row['x0'], row['y1'] - row['y0']
    x, y = row['x0'], row['y0']
    prob_ix = int(row['proba']*10)-7
    col = ['yellow', 'orange', 'red'][prob_ix]
    return plt.Rectangle((x, y), w, h, color=col, fill=False, lw=2)

def create_rect_raw(row):
    w, h = row['x1'] - row['x0'], row['y1'] - row['y0']
    x, y = row['x0'], row['y0']
    return plt.Rectangle((x, y), w, h, color='red', fill=False, lw=2)

rfcn = pd.read_csv('../features/comp4_det_additional_lesion.txt', sep = ' ', header = None,\
        names = ['img', 'proba', 'x0', 'y0', 'x1', 'y1'])
rfcn = rfcn[rfcn['proba']>0.9].reset_index(drop=True)

# Number of test files
print(rfcn[rfcn.img.str.contains('test')]['img'].unique().shape)
print(rfcn[rfcn.img.str.contains('test')].shape)

# 
rfcn = rfcn.loc[rfcn.groupby('img').proba.idxmax()]

if validate:
    samps = range(20)
    for c, row in rfcn.iloc[samps].iterrows():
        img = imread(DATADIR + '/%s.jpg'%(row['img']))
        plt.figure(figsize=(8,6))
        plt.imshow(img)
        plt.gca().add_patch(create_rect(row))

# Add the manual boundary box values
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
bboxdf['x1'] = bboxdf['x0'] + bboxdf['w']
bboxdf['y1'] = bboxdf['y0'] + bboxdf['h']
bboxdf['img'] = 'train/' + bboxdf['img']
rfcn['img'] = rfcn['img'] + '.jpg'

rfcn['w'] = rfcn['x1'] - rfcn['x0']
rfcn['h'] = rfcn['y1'] - rfcn['y0']

rfcn['annotation_type'] = 'prediction'
bboxdf['annotation_type'] = 'manual'
cols = bboxdf.columns.tolist()
bbox = pd.concat([bboxdf[cols], rfcn[cols]], axis = 0)
bbox.to_csv("../features/rfcn_predictions_and_manual_annotations.csv", index= None)
bbox = bbox.drop(['annotation_type'], axis=1)

# Load theimage sizes
img_sizes = []
for c, row in bbox.iterrows():
    img_sizes.append(list(Image.open(open(DATADIR + '/' + row['img'])).size))
img_sizes = pd.DataFrame(img_sizes, columns=['img_w', 'img_h'])
bbox['img_w'], bbox['img_h'] = img_sizes['img_w'], img_sizes['img_h']

# Make the boundary box square and not fall over the edge of the image
bbox['dim'] = bbox[['h', 'w']].max(axis=1).clip(lower=MIN_DIMENSION)
bbox['dim'] = bbox[['img_h', 'img_w', 'dim']].min(axis=1)
bbox['h_diff'] = bbox['dim'] - bbox['h']
bbox['w_diff'] = bbox['dim'] - bbox['w']
bbox[bbox['h_diff']<0]['h_diff'] = 0
bbox[bbox['w_diff']<0]['w_diff'] = 0
bbox['x0'] = bbox['x0'] - bbox['w_diff'].divide(2)
bbox['x1'] = bbox['x1'] + bbox['w_diff'].divide(2)
bbox['y0'] = bbox['y0'] - bbox['h_diff'].divide(2)
bbox['y1'] = bbox['y1'] + bbox['h_diff'].divide(2)
bbox['w'] = bbox['x1'] - bbox['x0'] 
bbox['h'] = bbox['y1'] - bbox['y0'] 
bbox[['x0', 'x1']] = bbox[['x0', 'x1']].add(np.where(bbox['x0']<0, bbox['x0'].abs(), 0), axis = 0 )
bbox[['y0', 'y1']] = bbox[['y0', 'y1']].add(np.where(bbox['y0']<0, bbox['y0'].abs(), 0), axis = 0 )
bbox[['x0', 'x1']] = bbox[['x0', 'x1']].subtract(np.where(bbox['x1']>bbox['img_w'], (bbox['x1']-bbox['img_w']).abs(), 0), axis = 0 )
bbox[['y0', 'y1']] = bbox[['y0', 'y1']].subtract(np.where(bbox['y1']>bbox['img_h'], (bbox['y1']-bbox['img_h']).abs(), 0), axis = 0 )
bbox['w'] = bbox['x1'] - bbox['x0'] 
bbox['h'] = bbox['y1'] - bbox['y0'] 
bbox['w']-bbox['h']


if validate:
    samps = range(4000,4010)
    for c, row in bbox.iloc[samps].iterrows():
        img = imread(DATADIR + '/%s'%(row['img']))
        plt.figure(figsize=(8,6))
        plt.imshow(img)
        plt.gca().add_patch(create_rect_raw(row))

if SAVE_BBOX:
    # Create the train val datasets
    if not os.path.exists('../data/rfcn_crop'):
        os.mkdir('../data/rfcn_crop')
        os.mkdir('../data/rfcn_crop/train')
        os.mkdir('../data/rfcn_crop/test')
        for i in range(3): os.mkdir('../data/rfcn_crop/Type_'+str(i+1))
        for i in range(3): os.mkdir('../data/rfcn_crop/train/Type_'+str(i+1))
    for c, row in bbox.iterrows():
        if c % 100 == 0 :print c
        img = imread(DATADIR + '/%s'%(row['img'])) 
        try:
            img = img[int(row['y0']):int(row['y1']), int(row['x0']):int(row['x1'])] # crop it in numpy direct
            imsave(DATACROPDIR + '/%s'%(row['img']), img)
        except:
            pass
        