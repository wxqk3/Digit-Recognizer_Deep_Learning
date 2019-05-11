import numpy as np
import pandas as pd
from numpy import *
from sklearn.model_selection import train_test_split
from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPool2D


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


x_train, y_train = prep_data(raw_data)
#to show the format of the input
print(shape(x_train),shape(y_train))



#part 2 buld a model-------------------------------------------------
#Create a `Sequential` model called `digitmodel`.
digitmodel = Sequential()

#first layer: Conv2D with 20 filters,3 x 3 kernal, relu activation function
digitmodel.add(Conv2D(20, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(img_rows, img_cols, 1)))

#second layer Conv2D,strides=2 to make it faster
digitmodel.add(Conv2D(20, kernel_size=(3, 3), activation='relu'))

#improvement

digitmodel.add(MaxPool2D(pool_size=(2,2)))
digitmodel.add(Dropout(0.25))



digitmodel.add(Conv2D(40, kernel_size=(3, 3), activation='relu'))
digitmodel.add(Conv2D(40, kernel_size=(3, 3), activation='relu'))
digitmodel.add(MaxPool2D(pool_size=(2,2)))
digitmodel.add(Dropout(0.25))





#3rd layer Flatten layer
digitmodel.add(Flatten())

#4th layer Dense layer with 128 neurons
digitmodel.add(Dense(128, activation='relu'))

#improvement

digitmodel.add(Dropout(0.5))

#5th layer prediction layer ,Dense layer softmax activation function
digitmodel.add(Dense(num_classes, activation='softmax'))



digitmodel.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])
digitmodel.fit(x_train, y_train,
          batch_size=128,
          epochs=2,
          validation_split = 0.2)

#predict the testing results
test_file = "test.csv"
test_data = pd.read_csv(test_file)

x_as_array = test_data.values[:,:]

num_images_test = test_data.shape[0]
x_shaped_array = x_as_array.reshape(num_images_test, img_rows, img_cols, 1)

x_test = x_shaped_array / 255

predictions = digitmodel.predict(x_test)
#most_likely_labels = decode_predictions(predictions, top=3)
print(shape(predictions))

#define output
output=[[0 for x in range(2)] for y in range(28000)]
count = 1
#array=np.array(predictions)
for row in output:
    row[0] = count
    count = count+1
    row[1] = np.argmax(predictions[count-2])



dataframe = pd.DataFrame(output,columns=['ImageId', 'Label'])
dataframe.to_csv("results.csv", index=False, sep=',')

#compare the accuarcy
#upload to kaggle , also could compare with standard.csv