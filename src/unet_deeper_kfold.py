
# coding: utf-8

# In[ ]:


import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import sys
import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import seaborn as sns
sns.set_style("white")

from sklearn.model_selection import train_test_split, StratifiedKFold

from tqdm import tqdm #, tnrange
#from itertools import chain
from skimage.io import imread, imshow #, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

import keras
from keras.models import Model, load_model, save_model
from keras.layers import Input,Dropout,BatchNormalization,Activation,Add
from keras.layers.core import Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K
from keras import optimizers

import tensorflow as tf

from keras.preprocessing.image import array_to_img, img_to_array, load_img#,save_img

from albumentations import (
    PadIfNeeded,
    HorizontalFlip,
    VerticalFlip,    
    CenterCrop,    
    Crop,
    Compose,
    Transpose,
    RandomRotate90,
    ElasticTransform,
    GridDistortion, 
    OpticalDistortion,
    RandomSizedCrop,
    OneOf,
    CLAHE,
    RandomContrast,
    RandomGamma,
    RandomBrightness
)

import time
from kaggle_util import *
from models import *

t_start = time.time()


# In[ ]:


img_size_ori = 101
img_size_target = 101

def upsample(img):
    if img_size_ori == img_size_target:
        return img
    return resize(img, (img_size_target, img_size_target), mode='constant', preserve_range=True)
    
def downsample(img):
    if img_size_ori == img_size_target:
        return img
    return resize(img, (img_size_ori, img_size_ori), mode='constant', preserve_range=True)

def build_model_deeper(input_layer, start_neurons, DropoutRatio = 0.5):
    # 101 -> 50
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(input_layer)
    conv1 = residual_block(conv1,start_neurons * 1)
    conv1 = residual_block(conv1,start_neurons * 1)
    conv1 = residual_block(conv1,start_neurons * 1, True)
    pool1 = MaxPooling2D((2, 2))(conv1)
    pool1 = Dropout(DropoutRatio/2)(pool1)

    # 50 -> 25
    conv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(pool1)
    conv2 = residual_block(conv2,start_neurons * 2)
    conv2 = residual_block(conv2,start_neurons * 2)
    conv2 = residual_block(conv2,start_neurons * 2, True)
    pool2 = MaxPooling2D((2, 2))(conv2)
    pool2 = Dropout(DropoutRatio)(pool2)

    # 25 -> 12
    conv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(pool2)
    conv3 = residual_block(conv3,start_neurons * 4)
    conv3 = residual_block(conv3,start_neurons * 4)
    conv3 = residual_block(conv3,start_neurons * 4, True)
    pool3 = MaxPooling2D((2, 2))(conv3)
    pool3 = Dropout(DropoutRatio)(pool3)

    # 12 -> 6
    conv4 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same")(pool3)
    conv4 = residual_block(conv4,start_neurons * 8)
    conv4 = residual_block(conv4,start_neurons * 8)
    conv4 = residual_block(conv4,start_neurons * 8, True)
    pool4 = MaxPooling2D((2, 2))(conv4)
    pool4 = Dropout(DropoutRatio)(pool4)

    # Middle
    convm = Conv2D(start_neurons * 16, (3, 3), activation=None, padding="same")(pool4)
    convm = residual_block(convm,start_neurons * 16)
    convm = residual_block(convm,start_neurons * 16)
    convm = residual_block(convm,start_neurons * 16, True)
    
    # 6 -> 12
    deconv4 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Dropout(DropoutRatio)(uconv4)
    
    uconv4 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same")(uconv4)
    uconv4 = residual_block(uconv4,start_neurons * 8)
    uconv4 = residual_block(uconv4,start_neurons * 8)
    uconv4 = residual_block(uconv4,start_neurons * 8, True)
    
    # 12 -> 25
    #deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
    deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="valid")(uconv4)
    uconv3 = concatenate([deconv3, conv3])    
    uconv3 = Dropout(DropoutRatio)(uconv3)
    
    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(uconv3)
    uconv3 = residual_block(uconv3,start_neurons * 4)
    uconv3 = residual_block(uconv3,start_neurons * 4)
    uconv3 = residual_block(uconv3,start_neurons * 4, True)

    # 25 -> 50
    deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = concatenate([deconv2, conv2])
        
    uconv2 = Dropout(DropoutRatio)(uconv2)
    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(uconv2)
    uconv2 = residual_block(uconv2,start_neurons * 2)
    uconv2 = residual_block(uconv2,start_neurons * 2)
    uconv2 = residual_block(uconv2,start_neurons * 2, True)
    
    # 50 -> 101
    #deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="valid")(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    
    uconv1 = Dropout(DropoutRatio)(uconv1)
    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(uconv1)
    uconv1 = residual_block(uconv1,start_neurons * 1)
    uconv1 = residual_block(uconv1,start_neurons * 1)
    uconv1 = residual_block(uconv1,start_neurons * 1, True)
    
    #uconv1 = Dropout(DropoutRatio/2)(uconv1)
    #output_layer = Conv2D(1, (1,1), padding="same", activation="sigmoid")(uconv1)
    output_layer_noActi = Conv2D(1, (1,1), padding="same", activation=None)(uconv1)
    output_layer =  Activation('sigmoid')(output_layer_noActi)
    
    return output_layer


# In[ ]:




# Loading of training/testing ids and depths
train_df = pd.read_csv("../input/train.csv", index_col="id", usecols=[0])
depths_df = pd.read_csv("../input/depths.csv", index_col="id")
train_df = train_df.join(depths_df)
test_df = depths_df[~depths_df.index.isin(train_df.index)]

len(train_df)

train_df["images"] = [np.array(load_img("../input/train/images/{}.png".format(idx), grayscale=True)) / 255 for idx in tqdm(train_df.index)]
train_df["masks"] = [np.array(load_img("../input/train/masks/{}.png".format(idx), grayscale=True)) / 255 for idx in tqdm(train_df.index)]
train_df["coverage"] = train_df.masks.map(np.sum) / pow(img_size_ori, 2)
train_df["coverage_class"] = train_df.coverage.map(cov_to_class)
train_df['hassalt'] = train_df['masks'].apply(lambda x: (x.max()!=0) * 1)


# In[ ]:


def get_splits(train_df, train_idx, val_idx):
    X_train = train_df.iloc[train_idx]
    X_valid = train_df.iloc[val_idx]
    x_train = np.array(X_train.images.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1)
    x_valid = np.array(X_valid.images.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1)
    msk_train = np.array(X_train.masks.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1)
    msk_val = np.array(X_valid.masks.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1)
    y_train = X_train.hassalt.values
    y_valid = X_valid.hassalt.values
    id_train = X_train.index.values
    id_valid = X_valid.index.values
    return x_train, x_valid, msk_train, msk_val, y_train, y_valid, id_train, id_valid

def argument(x_train, msk_train, y_train):
    aug_img = []
    aug_msk = []
    aug_y = []
    augments = [
        (1, HorizontalFlip(p=1)),
        (0.25, VerticalFlip(p=1)),
        (0.25, RandomRotate90(p=1)),
        (0.25, Transpose(p=1)),
        (0.25, ElasticTransform(p=1, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03)),
        (0.25, GridDistortion(p=1)),
        (0.25, OpticalDistortion(p=1, distort_limit=2, shift_limit=0.5)),
#         (0.5, RandomSizedCrop(p=1, min_max_height=(int(img_size_ori / 2), img_size_ori), height=img_size_ori, width=img_size_ori)),
    ]

    for ratio, aug in tqdm(augments):
        selidx = np.random.choice(x_train.shape[0], int(x_train.shape[0] * ratio), replace=False)
        for idx in tqdm(selidx):
            augmented = aug(image=x_train[idx], mask=msk_train[idx])
            aimg = augmented['image']
            amsk = augmented['mask']
            if len(aimg.shape) < 3:
                aimg = aimg[...,np.newaxis]
            if len(amsk.shape) < 3:
                amsk = amsk[...,np.newaxis]
            aug_img.append(aimg)
            aug_msk.append(amsk)
            aug_y.append(y_train[idx])

    aug_img = np.asarray(aug_img)
    aug_msk = np.asarray(aug_msk)
    aug_y = np.asarray(aug_y)
    x_train = np.append(x_train, aug_img, axis=0)
    msk_train = np.append(msk_train, aug_msk, axis=0)
    y_train = np.append(y_train, aug_y, axis=0)
    print(x_train.shape)
    print(msk_train.shape)
    print(y_train.shape)
    
    return x_train, msk_train, y_train


# In[ ]:


def train_model(fold, x_train, y_train, x_valid, y_valid, id_valid):
    start_feature = 32
    batch_size = 32
    dropout = 0.5
    base_name = 'Unet_resnet_deeper_{}_{}_{}_{}'.format(start_feature, batch_size, dropout, fold)
    save_model_name = '../model/segmenter/{}.model'.format(base_name)
    submission_dir = '../result/segmenter/{}.csv'.format(base_name)
    oof_dir = '../result/segmenter_oof/{}'.format(base_name)

    print(save_model_name)
    print(submission_dir)
    print(oof_dir)

    # model
    input_layer = Input((img_size_target, img_size_target, 1))
    output_layer = build_model_deeper(input_layer, start_feature,dropout)

    model1 = Model(input_layer, output_layer)

    c = optimizers.adam(lr = 0.01)
    model1.compile(loss="binary_crossentropy", optimizer=c, metrics=[my_iou_metric])

    board = keras.callbacks.TensorBoard(log_dir='log/segmenter/{}'.format(base_name),
                           histogram_freq=0, write_graph=True, write_images=False)
    early_stopping = EarlyStopping(monitor='val_my_iou_metric', mode = 'max',patience=10, verbose=1)
    model_checkpoint = ModelCheckpoint(save_model_name,monitor='val_my_iou_metric', 
                                       mode = 'max', save_best_only=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_my_iou_metric', mode = 'max',factor=0.5, patience=3, min_lr=0.00001, verbose=1)

    epochs = 200

    history = model1.fit(x_train, y_train,
                        validation_data=[x_valid, y_valid], 
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=[board, early_stopping, model_checkpoint,reduce_lr], 
                        verbose=1)
    
    model1 = load_model(save_model_name,custom_objects={'my_iou_metric': my_iou_metric})
    # remove layter activation layer and use losvasz loss
    input_x = model1.layers[0].input
    output_layer = model1.layers[-1].input
    model = Model(input_x, output_layer)
    c = optimizers.adam(lr = 0.01)

    # lovasz_loss need input range (-∞，+∞), so cancel the last "sigmoid" activation  
    # Then the default threshod for pixel prediction is 0 instead of 0.5, as in my_iou_metric_2.
    model.compile(loss=lovasz_loss, optimizer=c, metrics=[my_iou_metric_2])

    #model.summary()
    early_stopping = EarlyStopping(monitor='val_my_iou_metric_2', mode = 'max',patience=20, verbose=1)
    model_checkpoint = ModelCheckpoint(save_model_name,monitor='val_my_iou_metric_2', 
                                       mode = 'max', save_best_only=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_my_iou_metric_2', mode = 'max',factor=0.5, patience=5, min_lr=0.00001, verbose=1)
    epochs = 200

    history = model.fit(x_train, y_train,
                        validation_data=[x_valid, y_valid], 
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=[board, model_checkpoint,reduce_lr,early_stopping], 
                        verbose=1)


# In[ ]:


skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1234)
for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df['hassalt'])):
    x_train, x_valid, msk_train, msk_val, y_train, y_valid, id_train, id_valid = get_splits(train_df, train_idx, val_idx)
    x_train, msk_train, y_train = argument(x_train, msk_train, y_train)
    
    model = train_model(fold, x_train, msk_train, x_valid, msk_val, id_valid)
    
    from keras import backend as K
    K.clear_session()

