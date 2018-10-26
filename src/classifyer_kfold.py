
# coding: utf-8

# In[ ]:


import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import sys
import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import seaborn as sns
sns.set_style("white")

# %matplotlib inline

from sklearn.model_selection import train_test_split, StratifiedKFold

from tqdm import tqdm #, tnrange
from skimage.io import imread, imshow #, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

import keras
from keras.models import Model, load_model, save_model
from keras.layers import Input,Dropout,BatchNormalization,Activation,Add, Dense, Input, Dropout
from keras.layers.core import Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K
from keras import optimizers

import tensorflow as tf

from keras.preprocessing.image import array_to_img, img_to_array, load_img#,save_img

import time
from kaggle_util import *
from models import *

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

def get_model(BASE_MODEL):
    from keras.preprocessing.image import ImageDataGenerator
    if BASE_MODEL=='VGG16':
        from keras.applications.vgg16 import VGG16 as PTModel, preprocess_input
    elif BASE_MODEL=='VGG19':
        from keras.applications.vgg19 import VGG19 as PTModel, preprocess_input
    elif BASE_MODEL=='RESNET52':
        from keras.applications.resnet50 import ResNet50 as PTModel, preprocess_input
    elif BASE_MODEL=='InceptionV3':
        from keras.applications.inception_v3 import InceptionV3 as PTModel, preprocess_input
    elif BASE_MODEL=='Xception':
        from keras.applications.xception import Xception as PTModel, preprocess_input
    elif BASE_MODEL=='DenseNet169': 
        from keras.applications.densenet import DenseNet169 as PTModel, preprocess_input
    elif BASE_MODEL=='DenseNet121':
        from keras.applications.densenet import DenseNet121 as PTModel, preprocess_input
    elif BASE_MODEL=='InceptionResNetV2':
        from keras.applications.inception_resnet_v2 import InceptionResNetV2 as PTModel, preprocess_input
    else:
        raise ValueError('Unknown model: {}'.format(BASE_MODEL))
        
    return PTModel, preprocess_input


# In[ ]:


# Loading of training/testing ids and depths
train_df = pd.read_csv("../input/train.csv", index_col="id", usecols=[0])
depths_df = pd.read_csv("../input/depths.csv", index_col="id")
train_df = train_df.join(depths_df)
test_df = depths_df[~depths_df.index.isin(train_df.index)]

len(train_df)

train_df["images"] = [np.array(load_img("../input/train/images/{}.png".format(idx), grayscale=False)) for idx in tqdm(train_df.index)]
train_df["masks"] = [np.array(load_img("../input/train/masks/{}.png".format(idx), grayscale=True)) / 255 for idx in tqdm(train_df.index)]
train_df["coverage"] = train_df.masks.map(np.sum) / pow(img_size_ori, 2)
train_df["coverage_class"] = train_df.coverage.map(cov_to_class)
train_df['hassalt'] = train_df['masks'].apply(lambda x: (x.max()!=0) * 1)


# In[ ]:


def get_splits(train_df, train_idx, val_idx):
    X_train = train_df.iloc[train_idx]
    X_valid = train_df.iloc[val_idx]
    x_train = np.array(X_train.images.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 3)
    x_valid = np.array(X_valid.images.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 3)
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

def train_classifier(fold, model_name, x_train, y_train, x_valid, y_valid, id_valid):
    print(model_name)
    x_train_act = x_train
    x_val_act = x_valid
#     x_train_act = x_train.astype(np.float32)
#     for idx in range(len(x_train_act)):
#         x_train_act[idx] = preprocess_input(x_train_act[idx])
#     print(x_train_act.max())

#     x_val_act = x_valid.astype(np.float32)
#     for idx in range(len(x_val_act)):
#         x_val_act[idx] = preprocess_input(x_val_act[idx])

    inputshape = x_train_act.shape[1:]
    PTModel, preprocess_input = get_model(model_name)
    base_pretrained_model = PTModel(input_shape = inputshape, 
                                  include_top = False, weights = 'imagenet')
    base_pretrained_model.trainable = False

    from keras import models, layers
    from keras.optimizers import Adam
    img_in = layers.Input(inputshape, name='Image_RGB_In')
    x = base_pretrained_model(img_in)
    x = layers.Flatten(name='flatten')(x)

    x = Dense(256)(x)
    x = BatchActivate(x)
    x = Dropout(0.5)(x)

    x = Dense(64)(x)
    x = BatchActivate(x)
    x = Dropout(0.5)(x)

    out_layer = layers.Dense(1, activation = 'sigmoid')(x)
    class_model = models.Model(inputs = [img_in], outputs = [out_layer], name = 'full_model')

    class_model.compile(optimizer = Adam(lr=0.01), 
                       loss = 'binary_crossentropy',
                       metrics = ['binary_accuracy'])

    batch_size = 32
    base_name = '{}_{}'.format(model_name, fold)
    save_model_name = '../model/classifier/{}.model'.format(base_name)
    submission_file = '../result/classifier/{}.csv'.format(base_name)
    oof_file = '../result/classifier/{}_oof.csv'.format(base_name)

    print(save_model_name)
    print(submission_file)
    print(oof_file)
    
    board = keras.callbacks.TensorBoard(log_dir='log/classifier/{}'.format(base_name),
                           histogram_freq=0, write_graph=True, write_images=False)
    early_stopping = EarlyStopping(monitor='val_binary_accuracy', mode = 'max',patience=5, verbose=1)
    model_checkpoint = ModelCheckpoint(save_model_name,monitor='val_binary_accuracy', 
                                       mode = 'max', save_best_only=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_binary_accuracy', mode = 'max',factor=0.5, patience=2, min_lr=0.00001, verbose=1)

    epochs = 200

    history = class_model.fit(x_train, y_train,
                        validation_data=[x_valid, y_valid], 
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=[board, early_stopping, reduce_lr, model_checkpoint],
                        verbose=1)
    
    model = load_model(save_model_name)
    
    oof = model.predict(x_valid)
    df_oof = pd.DataFrame()
    df_oof['id'] = id_valid
    df_oof['target'] = oof
    df_oof.to_csv(oof_file, index=False)
    
    files = os.listdir('../input/test/images/')
    x_test = np.array([(np.array(load_img("../input/test/images/{}".format(idx), grayscale = False))) for idx in files]).reshape(-1, img_size_target, img_size_target, 3)
    preds_test = model.predict(x_test)
    df_result = pd.DataFrame()
    df_result['id'] = files
    df_result['pre'] = preds_test.reshape(len(files))
    df_result.to_csv(submission_file, index=False)

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1234)
for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df['hassalt'])):
    x_train, x_valid, msk_train, msk_val, y_train, y_valid, id_train, id_valid = get_splits(train_df, train_idx, val_idx)
    x_train, msk_train, y_train = argument(x_train, msk_train, y_train)
    
    model = train_classifier(fold, 'VGG16', x_train, y_train, x_valid, y_valid, id_valid)
    
    from keras import backend as K
    K.clear_session()