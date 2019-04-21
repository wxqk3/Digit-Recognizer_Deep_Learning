import pandas as pd
import matplotlib.pyplot as plt
from numpy import *

def opencsv():  # open with pandas
    data = pd.read_csv('train.csv')
    data1 = pd.read_csv('test.csv')
    train_data = data.values[0:, 1:]

    train_label = data.values[0:, 0]
    test_data = data1.values[0:, 0:]
    print ('Data Load Done!')
    return train_data, train_label, test_data

train_data, train_label, test_data = opencsv()


data = pd.read_csv('train.csv')
data2 = pd.read_csv('train.csv')
print (shape(train_data),shape(test_data))

#show the first five images
def showPic(data):
    for i in range(5):
        pixels = train_data[i]
        pixels = pixels.reshape((28, 28))
        print(shape(pixels))
        label=train_label[i]
        plt.title('Label is {label}'.format(label=label))
        plt.imshow(pixels, cmap='gray')
        plt.show()



showPic(train_data)
print('show pic Done!')
