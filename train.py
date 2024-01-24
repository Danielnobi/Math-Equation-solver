print("starting")
import pandas as pd
import numpy as np
import pickle

df_train=pd.read_csv('train_final.csv',index_col=False)
labels=df_train[['label']]  # Indexing is zero-based, so 783 corresponds to the 784th column
df_train = df_train.drop('label', axis=1)

np.random.seed(1212)
import keras
from keras.models import Model
from keras.layers import *
from keras import optimizers
from keras.layers import Input, Dense
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
# K.set_image_dim_ordering('th')

print("Done")

labels=np.array(labels)

from keras.utils.np_utils import to_categorical
cat=to_categorical(labels,num_classes=19)

l=[]
for i in range(50443):
    l.append(np.array(df_train[i:i+1]).reshape(28,28,1))

print("Input data shape:", np.array(l).shape)
print("Labels shape:", cat.shape)

# Ensure both have the same number of samples
assert np.array(l).shape[0] == cat.shape[0], "Number of samples mismatch between input data and labels"

np.random.seed(7)

model = Sequential()
model.add(Conv2D(30, (5, 5), input_shape=(28,28,1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(15, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(19, activation='softmax'))
# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from keras.models import model_from_json

model.fit(np.array(l), cat, epochs=10, batch_size=200,shuffle=True,verbose=1)

model_json = model.to_json()
with open("model_final.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_files/model_final2.h5")