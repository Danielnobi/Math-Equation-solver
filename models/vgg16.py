print("Loading...")

# common libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
# %matplotlib inline
import os
from os import listdir
from os.path import isfile, join

# CV and Image
import cv2
from PIL import Image

# pickle
import pickle

# keras
import keras
from keras import optimizers
from keras import backend as K
from keras.models import Model, Sequential, load_model
from keras.layers import *
from keras.layers import Input, Dense, Dropout, Flatten
from keras.utils import np_utils
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils.np_utils import to_categorical
from keras.utils.vis_utils import plot_model
# from tensorflow.keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.optimizers import adam_v2
K.image_data_format()

print("Done")

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2
# from google.colab.patches import cv2_imshowh
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras import backend as K
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

#ignore warning messages 
import warnings
warnings.filterwarnings('ignore') 

sns.set()

import joblib
from joblib import Parallel, delayed
from sklearn.tree import BaseDecisionTree

import torch
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel

import pathlib
dataset_url = f'./dataset'
# data_dir = tf.keras.utils.get_file('dataset', origin=dataset_url)
data_dir = pathlib.Path(dataset_url)

# from keras.utils import to_categorical

# # Assuming your original labels are in train_labels
# data_dir = to_categorical(data_dir, num_classes=19)
# print(data_dir)

img_height,img_width=28,28
batch_size=32
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

# from keras.utils import to_categorical

# # Assuming your original labels are in train_labels
# train_ds = to_categorical(train_ds, num_classes=19)

# class_names = train_ds.class_names
# print(class_names)
img_height, img_width = 32, 32

# # Resize images in train_ds
from keras.utils import to_categorical

# Assuming your original labels are in train_labels
train_ds = train_ds.map(lambda x, y: (tf.image.resize(x, (img_height, img_width)), tf.one_hot(y, depth=19)))

# Assuming your original labels are in val_labels
val_ds = val_ds.map(lambda x, y: (tf.image.resize(x, (img_height, img_width)), tf.one_hot(y, depth=19)))

resnet_model = Sequential()

pretrained_model= tf.keras.applications.ResNet50(include_top=False,
                   input_shape=(32,32,3),
                   pooling='avg',classes=19,
                   weights='imagenet')
for layer in pretrained_model.layers:
        layer.trainable=False

resnet_model.add(pretrained_model)
resnet_model.add(Flatten())
resnet_model.add(Dense(512, activation='relu'))
resnet_model.add(Dense(19, activation='softmax'))

resnet_model.compile(optimizer=Adam(lr=0.002),loss='categorical_crossentropy',metrics=['accuracy'])

resnet_model.fit(train_ds, validation_data=val_ds, epochs=20)

tf.keras.models.save_model(resnet_model, f'./Model Files/vgg16.h5', save_format='h5')