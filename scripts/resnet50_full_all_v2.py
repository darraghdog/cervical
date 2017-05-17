
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import gc, math
import pickle

from keras.models import Sequential
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.models import Model, load_model, model_from_json

from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers.normalization import BatchNormalization
from keras import regularizers
from keras import backend as K
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard

from sklearn.metrics import log_loss, accuracy_score, confusion_matrix

from cnnmodels5 import vgg_std16_model, preprocess_input, create_rect5, load_img, train_generator, test_generator
from cnnmodels5 import identity_block, testcv_generator, conv_block, resnet50_model, save_array, load_array


# In[25]:

# Params
img_rows, img_cols = 224, 224 # Resolution of inputs
channel = 3
ROWS, COLS = 224, 224
CHECKPOINT_DIR = 'log/checkpoint05/'
BATCHSIZE = 32
CERV_CLASSES = ['Type_1', 'Type_2', 'Type_3']
nb_perClass = int(BATCHSIZE / len(CERV_CLASSES))
TRAIN_DIRDM = '../data/cropped/train'
TEST_DIRDM = '../data/cropped/test'
TRAIN_DIRS = ['../data/original/train', '../data/rfcn_crop/train', '../data/gmm/train']
TEST_DIRS = ['../data/original/test', '../data/rfcn_crop/test', '../data/gmm/test']
DATA_DIRS = ['../data/original', '../data/rfcn_crop', '../data/gmm']
REMOVE_FOLDER = ['../', 'data/', 'cropped/', 'rfcn_crop/', 'gmm/',  'original/']
num_class = len(CERV_CLASSES)
full = True # False
bags = 5


# In[3]:

train_datagen = ImageDataGenerator(
    rotation_range=180,
    shear_range=0.2,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True)


# In[4]:

img_ls = []
y_ls = []
# Load Raw images in original directory structure format
for TRAIN_DIR in TRAIN_DIRS:
    for typ in CERV_CLASSES:
        for img in os.listdir(os.path.join(TRAIN_DIR, typ)):
            if img != '.DS_Store':
                img_ls.append(os.path.join(TRAIN_DIR, typ, img))
                y_ls.append(typ)
# Load rfcn cropped images in original directory structure format
for DATA_DIR in DATA_DIRS:
    for typ in CERV_CLASSES:
        for img in os.listdir(os.path.join(DATA_DIR, typ)):
            if img != '.DS_Store':
                img_ls.append(os.path.join(DATA_DIR, typ, img))
                y_ls.append(typ)
# Load Dave's crop images in simplified directory structure
for typ in CERV_CLASSES:
    for img in os.listdir(os.path.join(TRAIN_DIRDM, typ)):
        if img != '.DS_Store':
            img_ls.append(os.path.join(TRAIN_DIRDM, typ, img))
            y_ls.append(typ)
train_all  = pd.DataFrame({'class': y_ls, 'img': img_ls, })[['img', 'class']]


# In[5]:

img_ls = []
for TEST_DIR in TEST_DIRS:
    for img in os.listdir(TEST_DIR):
        if img != '.DS_Store':
            img_ls.append(os.path.join(TEST_DIR, img))
for img in os.listdir(TEST_DIRDM):
    if img != '.DS_Store':
        img_ls.append(os.path.join(TEST_DIRDM, img))
test_df  = pd.DataFrame({'img': img_ls}) 


# In[6]:

print(test_df.shape)
print(train_all.shape)
print(train_all.head(2))
print(train_all.tail(2))


# In[7]:

def train_generator(datagen, df):
    while 1:
        batch_x = np.zeros((BATCHSIZE, ROWS, COLS, 3), dtype=K.floatx())
        batch_y = np.zeros((BATCHSIZE, len(CERV_CLASSES)), dtype=K.floatx())
        fn = lambda obj: obj.loc[np.random.choice(obj.index, size=nb_perClass, replace=False),:]
        batch_df = df.groupby('class', as_index=True).apply(fn)
        i = 0
        for index,row in batch_df.iterrows():
            row = row.tolist()
            image_file = row[0]
            typ_class = row[1]
            img = Image.open(image_file).resize((ROWS, COLS))
            img = img.convert('RGB')
            x = np.asarray(img, dtype=K.floatx())
            #x = datagen.random_transform(x)
            x = preprocess_input(x)
            batch_x[i] = x
            batch_y[i,CERV_CLASSES.index(typ_class)] = 1
            i += 1
            #return (batch_x, batch_y)
        yield (batch_x.transpose(0, 3, 1, 2), batch_y)


# In[8]:

# Split into train and valid
valid_set = pd.read_csv("../val_images.csv", header = None, names = ['img']).img.tolist()
#validx = pd.read_csv("../cv_split.csv")
#valid_set = []
#test_fold = 0
#for c, row in validx.iterrows():
#    if row['fold'] == test_fold:
#        valid_set.append(row['SubDirectory']+'/'+row['file_name'])
#    else:
#        continue
valid_set[-4:]


# In[9]:

orig_format = []
for c, row in train_all.iterrows():
    if '../data/cropped' in row['img']:
        typ = row['img'].split('/')[-2]
        additional = True if 'additional' in row['img'] else False
        img = row['img'].split('/')[-1].split('_')[0]
        if additional:
            orig_format.append(typ+'/'+img+'.jpg')
        else:
            orig_format.append('train/'+typ+'/'+img+'.jpg')
    else:
        orig_nm = row['img']
        for data_dir in REMOVE_FOLDER:
            orig_nm = orig_nm.replace(data_dir, '')
        orig_format.append(orig_nm)
        
train_all['orig_format'] = orig_format


# In[10]:

train_all.tail()


# In[11]:

valid_df = train_all[train_all['orig_format'].isin(valid_set)]
if full == True:
    train_df = train_all
else:
    train_df = train_all[~train_all['orig_format'].isin(valid_set)]
samples_per_epoch=BATCHSIZE*math.ceil(train_df.groupby('class').size()['Type_2']/nb_perClass)
print(train_df.shape)
print(valid_df.shape)


# In[12]:

# Make our validation set
if os.path.exists("results/valid_x.dat"):
    valid_x = load_array('results/valid_x.dat')
    valid_y = load_array('results/valid_y.dat')
else:
    l = valid_df.groupby('class').size()
    valid_x = np.zeros((valid_df.shape[0], ROWS, COLS, 3), dtype=K.floatx())
    valid_y = np.zeros((valid_df.shape[0], len(CERV_CLASSES)), dtype=K.floatx())
    i = 0
    for index,row in valid_df.iterrows():
        if i % 1000 == 0 : print('Loading val image {} out of {}'.format(i, valid_df.shape[0]))
        row = row.tolist()
        image_file = row[0]
        typ_class = row[1]
        img = Image.open(image_file).resize((ROWS, COLS))
        img = img.convert('RGB')
        x = np.asarray(img, dtype=K.floatx())
        # x = datagen.random_transform(x)
        x = preprocess_input(x)
        valid_x[i] = x
        valid_y[i,CERV_CLASSES.index(typ_class)] = 1
        i += 1
    valid_x = valid_x.transpose(0, 3, 1, 2)
    save_array('results/valid_x.dat', valid_x)
    save_array('results/valid_y.dat', valid_y)


# In[13]:

def test_generator(df, datagen, batch_size = BATCHSIZE):
    n = df.shape[0]
    batch_index = 0
    while 1:
        current_index = batch_index * batch_size
        if n >= current_index + batch_size:
            current_batch_size = batch_size
            batch_index += 1    
        else:
            current_batch_size = n - current_index
            batch_index = 0        
        batch_df = df[current_index:current_index+current_batch_size]
        batch_x = np.zeros((batch_df.shape[0], ROWS, COLS, 3), dtype=K.floatx())
        i = 0
        for index,row in batch_df.iterrows():
            row = row.tolist()
            image_file = row[0]
            # typ_class = row[1]
            img = Image.open(image_file).resize((ROWS, COLS))
            img = img.convert('RGB')
            x = np.asarray(img, dtype=K.floatx())
            # x = datagen.random_transform(x)
            x = preprocess_input(x)
            batch_x[i] = x
            i += 1
        if batch_index%100 == 0: print(batch_index)
        # return (batch_x.transpose(0, 3, 1, 2))
        yield(batch_x.transpose(0, 3, 1, 2))


# In[14]:

early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')        
model_checkpoint = ModelCheckpoint(filepath=CHECKPOINT_DIR+'weights.{epoch:03d}-{val_loss:.4f}.hdf5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto')
# learningrate_schedule = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, mode='auto', epsilon=0.001, cooldown=0, min_lr=0)
# tensorboard = TensorBoard(log_dir=LOG_DIR, histogram_freq=0, write_graph=False, write_images=True)


# In[15]:

print "Model creation... "
nb_epoch = 2
model = resnet50_model(ROWS, COLS, channel, num_class)
for layer in model.layers:
    layer.trainable = False
for layer in model.layers[-3:]:
    layer.trainable = True


# model.optimizer.lr = 1e-5
# Start Fine-tuning
print "Fine tune part 1"
model.fit_generator(train_generator(train_datagen, train_df),
          nb_epoch=nb_epoch,
          samples_per_epoch=samples_per_epoch, #40864
          verbose=1,
          validation_data=(valid_x, valid_y),
          callbacks=[early_stopping, model_checkpoint],
          )


# In[23]:

# Start Fine-tuning
nb_epoch = 5
print "Fine tune part 1A"
model.fit_generator(train_generator(train_datagen, train_df),
          nb_epoch=nb_epoch,
          samples_per_epoch=samples_per_epoch, #50000,
          verbose=1,
          validation_data=(valid_x, valid_y),
          callbacks=[early_stopping, model_checkpoint],
          )


# In[24]:

### Resnet50
# fine tuning
# 164 conv5c+top
# 142 conv5+top
# 80 conv4+conv5+top
# 38 conv3+conv4+conv5+top
start_layer = 164

model.optimizer.lr = 1e-6
for layer in model.layers[start_layer:]:
    layer.trainable = True
nb_epoch = 8
print "Fine tune part 2"
model.fit_generator(train_generator(train_datagen, df=train_df),
          nb_epoch=nb_epoch,
          samples_per_epoch=samples_per_epoch,
          verbose=1,
          validation_data=(valid_x, valid_y),
          callbacks=[model_checkpoint, early_stopping], # , 
          )


# In[21]:

# Hack to solve issue on model loading : https://github.com/fchollet/keras/issues/4044
import glob
import h5py
import time
timestr = time.strftime("%Y%m%d")


model_files = sorted(glob.glob(CHECKPOINT_DIR + '*.hdf5'))
for model_file in model_files:
    print("Update '{}'".format(model_file))
    with h5py.File(model_file, 'a') as f:
        if 'optimizer_weights' in f.keys():
            del f['optimizer_weights']


# ### Bag the predictions from a few epochs

# In[ ]:

import glob
files = glob.glob(CHECKPOINT_DIR+'*')
val_losses = [float(f.split('-')[-1][:-5]) for f in files]
min_id = np.array(val_losses).argsort()[:bags].tolist()


# In[ ]:

# Loop the lowest val losses and get a prediction for each
test_preds_ls = []
for index in min_id:
    print('Loading model from checkpoints file ' + files[index])
    test_model = load_model(files[index])
    test_model_name = files[index].split('/')[-2][-1:]+'_'+files[index].split('/')[-1]
    test_preds_ls.append(test_model.predict_generator(test_generator(test_df, train_datagen), 
                                         val_samples = test_df.shape[0])) 
    del test_model
    gc.collect()


# In[ ]:

test_preds = sum(test_preds_ls)/len(test_preds_ls)


# In[ ]:

test_sub = pd.DataFrame(test_preds, columns=CERV_CLASSES)
test_sub['image_name'] = test_df['img'].str.split('/').apply(lambda x: x[-1])
test_sub = test_sub[['image_name'] + CERV_CLASSES ]
test_sub.head(3)


# In[ ]:

if full:
    subm_name = '../sub/sub_dara_full_resnet_all_5xbag_' + timestr + '.csv' #'.csv.gz'
else:
    subm_name = '../sub/sub_dara_part_resnet_all_5xbag_' + timestr + '.csv' #'.csv.gz'
    
#test_sub.to_csv(subm_name, index=False)#, compression='gzip')
#FileLink(subm_name)

