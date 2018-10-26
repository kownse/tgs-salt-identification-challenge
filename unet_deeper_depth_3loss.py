
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
tqdm_notebook = tqdm
#from itertools import chain
from skimage.io import imread, imshow #, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

import keras
from keras.models import Model, load_model, save_model
from keras.layers import Input,Dropout,BatchNormalization,Activation,Add
from keras.layers.core import Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K
from keras import optimizers

import tensorflow as tf

from keras.preprocessing.image import array_to_img, img_to_array, load_img#,save_img

import time
from kaggle_util import *
from models import *

t_start = time.time()


# In[2]:


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
    output_layer = Conv2D(1, (1,1), padding="same", activation='relu')(uconv1)
    
#     output_layer =  Activation('sigmoid')(output_layer_noActi)

    ga = GlobalAveragePooling2D()(convm)
    ga = Dropout(DropoutRatio)(ga)
    empty = Dense(256, kernel_initializer='he_uniform')(ga)
    empty = BatchActivate(empty)
    empty = Dropout(DropoutRatio)(empty)
    empty = Dense(64, kernel_initializer='he_uniform')(empty)
    empty = BatchActivate(empty)
    empty = Dropout(DropoutRatio)(empty)
    out_empty = Dense(1, activation='sigmoid', name='empty_out')(empty)
    
    final_out_noact = multiply([output_layer, out_empty])
    final_out_noact = BatchNormalization()(final_out_noact)
    final_out =  Activation('sigmoid', name = 'segment_out')(final_out_noact)
    
    return final_out, out_empty


# In[3]:


# Loading of training/testing ids and depths
train_df = pd.read_csv("../input/train.csv", index_col="id", usecols=[0])
depths_df = pd.read_csv("../input/depths.csv", index_col="id")
train_df = train_df.join(depths_df)
test_df = depths_df[~depths_df.index.isin(train_df.index)]

len(train_df)

train_df["images"] = [np.array(load_img("../input/train/images/{}.png".format(idx), grayscale=True)) / 255 for idx in tqdm_notebook(train_df.index)]
train_df["masks"] = [np.array(load_img("../input/train/masks/{}.png".format(idx), grayscale=True)) / 255 for idx in tqdm_notebook(train_df.index)]
train_df["coverage"] = train_df.masks.map(np.sum) / pow(img_size_ori, 2)
train_df["coverage_class"] = train_df.coverage.map(cov_to_class)
train_df['empty'] = train_df['masks'].apply(lambda x: (x.max()!=0) * 1)


# In[4]:


z_max = train_df['z'].max()
z_min = train_df['z'].min()
z_dis = z_max - z_min
train_df['z'] = (train_df['z'] - z_min) / z_dis
step = 1 / z_dis


# In[5]:


train_df['empty'].value_counts()


# In[6]:


SUBSET = len(train_df)
train_df = train_df.head(SUBSET)
len(train_df)


# In[7]:


ids_train, ids_valid, x_train, x_valid, y_train, y_valid, cov_train, cov_test, depth_train, depth_test, empty_train, empty_test = train_test_split(
train_df.index.values,
np.array(train_df.images.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target), 
np.array(train_df.masks.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1), 
train_df.coverage.values,
train_df.z.values,
train_df['empty'].values,
test_size=0.2, stratify=train_df.coverage_class, random_state= 1234)


# In[8]:


#Data augmentation
x_train = np.append(x_train, [np.fliplr(x) for x in x_train], axis=0)
y_train = np.append(y_train, [np.fliplr(x) for x in y_train], axis=0)
depth_train = np.append(depth_train, depth_train, axis=0)
empty_train = np.append(empty_train, empty_train, axis=0)

x_train = add_depth_bulk_act(x_train, depth_train, step)
x_valid = add_depth_bulk_act(x_valid, depth_test, step)
print(x_train.shape)
print(y_valid.shape)
print(empty_train.shape)


# In[9]:


start_feature = 32
batch_size = 32
dropout = 0.5
base_name = 'Unet_resnet_3loss_depth_{}_{}_{}'.format(start_feature, batch_size, dropout)
basic_name = '../model/{}'.format(base_name)
save_model_name = basic_name + '.model'
submission_file = basic_name + '.csv'

print(save_model_name)
print(submission_file)

# model
input_layer = Input((img_size_target, img_size_target, 3))
output_layer, out_empty = build_model_deeper(input_layer, start_feature,dropout)

model1 = Model(input_layer, [output_layer, out_empty])

losses = {
    'empty_out' : 'binary_crossentropy',
    'segment_out': 'binary_crossentropy',
    #'final_out' : lovasz_loss,
}
lossWeights = {
    'empty_out' : 0.2,
    'segment_out':2,
    #'final_out' : 3,
}
c = optimizers.adam(lr = 0.01)
model1.compile(loss=losses, loss_weights=lossWeights, optimizer=c, metrics=[my_iou_metric])


# In[10]:


y_combine_rain = {
    'empty_out' : empty_train,
    'segment_out':y_train,
    #'final_out' : y_train,
}

y_combine_test = {
    'empty_out' : empty_test,
    'segment_out':y_valid,
    #'final_out' : y_valid,
}

epochs = 200

board = keras.callbacks.TensorBoard(log_dir='log/{}'.format(base_name),
                       histogram_freq=0, write_graph=True, write_images=False)
early_stopping = EarlyStopping(monitor='val_segment_out_my_iou_metric', mode = 'max',patience=12, verbose=1)
model_checkpoint = ModelCheckpoint(save_model_name,monitor='val_segment_out_my_iou_metric', 
                                   mode = 'max', save_best_only=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_segment_out_my_iou_metric', mode = 'max',factor=0.5, patience=3, min_lr=0.00001, verbose=1)


history = model1.fit(x_train, y_combine_rain,
                    validation_data=[x_valid, y_combine_test], 
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[board, early_stopping, model_checkpoint,reduce_lr], 
                    verbose=1)


# In[ ]:


base_name = 'Unet_resnet_3loss_depth_{}_{}_{}'.format(start_feature, batch_size, dropout)
basic_name = '../model/{}'.format(base_name)
save_model_name = basic_name + '.model'

model1 = load_model(save_model_name,custom_objects={'my_iou_metric': my_iou_metric})


# In[ ]:



base_name = 'Unet_resnet_3loss_depth_stage2_{}_{}_{}'.format(start_feature, batch_size, dropout)
basic_name = '../model/{}'.format(base_name)
save_model_name = basic_name + '.model'
submission_file = basic_name + '.csv'

input_x = model1.layers[0].input
model1.layers[-1].name = 'segment_out_old'

output_layer = model1.layers[-2]
output_layer.name = 'segment_out'
output_layer = output_layer.output

empty_out = model1.get_layer("empty_out").output
model = Model(input_x, [output_layer, empty_out])
c = optimizers.adam(lr = 0.01)

losses = {
    'empty_out' : 'binary_crossentropy',
    'segment_out': lovasz_loss,
    #'final_out' : lovasz_loss,
}
lossWeights = {
    'empty_out' : 0.2,
    'segment_out':2,
    #'final_out' : 3,
}
model.compile(loss=losses, loss_weights=lossWeights, optimizer=c, metrics=[my_iou_metric_2])

early_stopping = EarlyStopping(monitor='val_segment_out_my_iou_metric_2', mode = 'max',patience=16, verbose=1)
model_checkpoint = ModelCheckpoint(save_model_name,monitor='val_segment_out_my_iou_metric_2', 
                                   mode = 'max', save_best_only=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_segment_out_my_iou_metric_2', mode = 'max',factor=0.5, patience=4, min_lr=0.00001, verbose=1)


history = model.fit(x_train, y_combine_rain,
                    validation_data=[x_valid, y_combine_test], 
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[board, early_stopping, model_checkpoint,reduce_lr], 
                    verbose=1)


# In[ ]:


model = load_model(save_model_name,custom_objects={'my_iou_metric_2': my_iou_metric_2,
                                                   'lovasz_loss': lovasz_loss})


# In[ ]:


preds_valid = predict_result(model,x_valid,img_size_target)
## Scoring for last model, choose threshold by validation data 
thresholds_ori = np.linspace(0.3, 0.7, 31)
# Reverse sigmoid function: Use code below because the  sigmoid activation was removed
thresholds = np.log(thresholds_ori/(1-thresholds_ori)) 

# ious = np.array([get_iou_vector(y_valid, preds_valid > threshold) for threshold in tqdm_notebook(thresholds)])
# print(ious)
ious = np.array([iou_metric_batch(y_valid, preds_valid > threshold) for threshold in tqdm_notebook(thresholds)])
print(ious)

# instead of using default 0 as threshold, use validation data to find the best threshold.
threshold_best_index = np.argmax(ious) 
iou_best = ious[threshold_best_index]
threshold_best = thresholds[threshold_best_index]

