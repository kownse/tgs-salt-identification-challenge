
# coding: utf-8

# In[1]:


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

from sklearn.model_selection import train_test_split

from tqdm import tqdm #, tnrange
#from itertools import chain
from skimage.io import imread, imshow #, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

import keras
from keras.models import Model, load_model, save_model
from keras.layers import Input,Dropout,BatchNormalization,Activation,Add
from keras.layers import Input, BatchNormalization, Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, concatenate, Concatenate, UpSampling2D, Activation
from keras.layers.core import Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K
from keras import optimizers
from keras import layers
from keras.applications.inception_resnet_v2 import InceptionResNetV2

import tensorflow as tf

from keras.preprocessing.image import array_to_img, img_to_array, load_img#,save_img

import time
from kaggle_util import *
from models import *
import cv2

t_start = time.time()

img_size_ori = 101
img_size_target = 128


# Loading of training/testing ids and depths
train_df = pd.read_csv("../input/train.csv", index_col="id", usecols=[0])
depths_df = pd.read_csv("../input/depths.csv", index_col="id")
train_df = train_df.join(depths_df)
test_df = depths_df[~depths_df.index.isin(train_df.index)]

len(train_df)

train_df["images"] = [np.array(load_img("../input/train/images/{}.png".format(idx), grayscale=False)) for idx in tqdm(train_df.index)]
train_df.images = train_df.images.apply(make_4)
train_df["masks"] = [np.array(load_img("../input/train/masks/{}.png".format(idx), grayscale=True)) / 255 for idx in tqdm(train_df.index)]
train_df["coverage"] = train_df.masks.map(np.sum) / pow(img_size_ori, 2)
train_df["coverage_class"] = train_df.coverage.map(cov_to_class)

SUBSET = len(train_df)
train_df = train_df.head(SUBSET)
len(train_df)

ids_train, ids_valid, x_train, x_valid, y_train, y_valid, cov_train, cov_test, depth_train, depth_test = train_test_split(
train_df.index.values,
np.array(train_df.images.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 4), 
np.array(train_df.masks.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1), 
train_df.coverage.values,
train_df.z.values,
test_size=0.2, stratify=train_df.coverage_class, random_state= 1234)


# In[6]:


#Data augmentation
x_train = np.append(x_train, [np.fliplr(x) for x in x_train], axis=0)
y_train = np.append(y_train, [np.fliplr(x) for x in y_train], axis=0)
print(x_train.shape)
print(y_valid.shape)


params = [
#     (0, 'None'),
#     (0, 'imagenet'),
    (0.1, 'None'),
#     (0.1, 'imagenet'),
]

for (dropout, weights) in params:
    print(dropout, weights)
    print()
    
    basic_name = '../model/inception_{}_{}'.format(dropout, weights)
    save_model_name = basic_name + '.model'
    submission_file = basic_name + '.csv'

    print(save_model_name)
    print(submission_file)
    
    input_layer = Input((img_size_target, img_size_target, 4))
    model1 = get_inception_resnet_v2_unet_sigmoid(input_layer,(img_size_target, img_size_target), 32,dropout, weights)

    c = optimizers.adam(lr = 0.01)
    model1.compile(loss="binary_crossentropy", optimizer=c, metrics=[my_iou_metric])

    board = keras.callbacks.TensorBoard(log_dir='log/inception_resunet_{}_{}'.format(dropout, weights),
                           histogram_freq=0, write_graph=True, write_images=False)
    early_stopping = EarlyStopping(monitor='val_my_iou_metric', mode = 'max',patience=10, verbose=1)
    model_checkpoint = ModelCheckpoint(save_model_name,monitor='val_my_iou_metric', 
                                       mode = 'max', save_best_only=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_my_iou_metric', mode = 'max',factor=0.5, patience=3, min_lr=0.00001, verbose=1)

    epochs = 200
    batch_size = 32
    history = model1.fit(x_train, y_train,
                        validation_data=[x_valid, y_valid], 
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=[board, early_stopping, model_checkpoint,reduce_lr], 
                        verbose=1)


    # In[8]:


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
    board = keras.callbacks.TensorBoard(log_dir='log/inception_resunet_{}_{}'.format(dropout, weights),
                           histogram_freq=0, write_graph=True, write_images=False)
    early_stopping = EarlyStopping(monitor='val_my_iou_metric_2', mode = 'max',patience=10, verbose=1)
    model_checkpoint = ModelCheckpoint(save_model_name,monitor='val_my_iou_metric_2', 
                                       mode = 'max', save_best_only=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_my_iou_metric_2', mode = 'max',factor=0.5, patience=3, min_lr=0.00001, verbose=1)
    epochs = 200
    batch_size = 32

    history = model.fit(x_train, y_train,
                        validation_data=[x_valid, y_valid], 
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=[board, model_checkpoint,reduce_lr,early_stopping], 
                        verbose=1)

    model = load_model(save_model_name,custom_objects={'my_iou_metric_2': my_iou_metric_2,
                                                       'lovasz_loss': lovasz_loss})

    preds_valid = predict_result(model,x_valid,img_size_target)
    ## Scoring for last model, choose threshold by validation data 
    thresholds_ori = np.linspace(0.3, 0.7, 31)
    # Reverse sigmoid function: Use code below because the  sigmoid activation was removed
    thresholds = np.log(thresholds_ori/(1-thresholds_ori)) 

    ious = np.array([iou_metric_batch(y_valid, preds_valid > threshold) for threshold in tqdm(thresholds)])
    print(ious)

    # instead of using default 0 as threshold, use validation data to find the best threshold.
    threshold_best_index = np.argmax(ious) 
    iou_best = ious[threshold_best_index]
    threshold_best = thresholds[threshold_best_index]

    x_test = [upsample(make_4(np.array(load_img("../input/test/images/{}.png".format(idx), grayscale=False)))) for idx in tqdm(test_df.index)]
    x_test = np.array(x_test).reshape(-1, img_size_target, img_size_target, 4)

    preds_test = predict_result(model,x_test,img_size_target)

    t1 = time.time()
    pred_dict = {idx: rle_encode(np.round(downsample(preds_test[i]) > threshold_best)) for i, idx in enumerate(tqdm(test_df.index.values))}
    t2 = time.time()

    print(f"Usedtime = {t2-t1} s")

    sub = pd.DataFrame.from_dict(pred_dict,orient='index')
    sub.index.names = ['id']
    sub.columns = ['rle_mask']
    sub = sub.reset_index()
    save_result(sub, '../result/' + submission_file, 
                            competition = 'tgs-salt-identification-challenge', 
                            send = True, index = False)

    t_finish = time.time()
    print(f"Kernel run time = {(t_finish-t_start)/3600} hours")

    from keras import backend as K
    K.clear_session()