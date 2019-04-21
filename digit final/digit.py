import numpy as np
import pandas as pd
from numpy import *
from sklearn.model_selection import train_test_split
from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout


img_rows, img_cols = 28, 28
num_classes = 10


#part 1 data preparation-------------------------------------------------
def prep_data(raw):
    #encode the label format , num_class=10, read the first column as label
    label_y = raw.values[:,0]
    out_y = keras.utils.to_categorical(label_y, num_classes)

    #encode train data(42000,784)
    num_images = raw.shape[0]
    x_as_array = raw.values[:,1:]

    #reshape to (42000,28,28,1)
    x_shaped_array = x_as_array.reshape(num_images, img_rows, img_cols, 1)

    #change to value from 0-255 to 0-1
    out_x = x_shaped_array / 255
    return out_x, out_y

train_file = "train.csv"
raw_data = pd.read_csv(train_file)


x, y = prep_data(raw_data)
#to show the format of the input
print(shape(x),shape(y))



#part 2 buld a model-------------------------------------------------
#Create a `Sequential` model called `digitmodel`.
digitmodel = Sequential()

#first layer: Conv2D with 20 filters,3 x 3 kernal, relu activation function
digitmodel.add(Conv2D(20, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(img_rows, img_cols, 1)))

#second layer Conv2D,strides=2 to make it faster
digitmodel.add(Conv2D(20, kernel_size=(3, 3), activation='relu', strides=2))

#3rd layer Flatten layer
digitmodel.add(Flatten())

#4th layer Dense layer with 128 neurons
digitmodel.add(Dense(128, activation='relu'))

#5th layer prediction layer ,Dense layer softmax activation function
digitmodel.add(Dense(num_classes, activation='softmax'))

digitmodel.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])
digitmodel.fit(x, y,
          batch_size=128,
          epochs=2,
          validation_split = 0.2)