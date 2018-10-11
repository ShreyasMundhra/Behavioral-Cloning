# Setup Keras
from scipy import ndimage
import csv
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Lambda, Cropping2D
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

from keras.models import load_model

def load_data():
    data_path = '/opt/carnd_p3/data/'
    df = pd.read_csv(data_path + 'driving_log.csv')
    lines = list(df.values)
            
    images = []
    measurements = []
    for i in range(1, len(lines)):
        line = lines[i]
        
        # get images from the three cameras
        image_center = ndimage.imread(data_path + 'IMG/' + line[0].split('/')[-1])
        image_left = ndimage.imread(data_path + 'IMG/' + line[1].split('/')[-1])
        image_right = ndimage.imread(data_path + 'IMG/' + line[2].split('/')[-1])
        
        # flip all images
        flipped_image_center = np.fliplr(image_center)
        flipped_image_left = np.fliplr(image_left)
        flipped_image_right = np.fliplr(image_right)
        
        # set steering angle for the three camera images
        steering_center = float(line[3])
        correction = 0.2 # this is a parameter to tune
        steering_left = steering_center + correction
        steering_right = steering_center - correction
        
        # set steering angle for the three flipped images
        flipped_steering_center = -steering_center
        flipped_steering_left = -steering_left
        flipped_steering_right = -steering_right
        
        images.extend([image_center, image_left, image_right, flipped_image_center, flipped_image_left, flipped_image_right])
        measurements.extend([steering_center, steering_left, steering_right, flipped_steering_center, flipped_steering_left, flipped_steering_right])
    
    X_train = np.array(images)
    y_train = np.array(measurements)
    return X_train, y_train

# load dataset
X_train, y_train = load_data()

# define the model architecture
model = Sequential()
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5))
model.add(Conv2D(filters=32, kernel_size=(3,3)))
model.add(MaxPooling2D())   # output size: [None, 44, 159, 32]
model.add(Activation('relu'))
model.add(Conv2D(filters=32, kernel_size=(3,3)))
model.add(MaxPooling2D())   # output size: [None, 21, 78, 32]
model.add(Activation('relu'))
model.add(Conv2D(filters=32, kernel_size=(3,3)))
model.add(MaxPooling2D())   # output size: [None, 9, 38, 32]
model.add(Dropout(rate=0.5))
model.add(Activation('relu'))
model.add(Flatten())    # output size: 10,944
model.add(Dense(5000))
model.add(Activation('relu'))
model.add(Dense(1000))
model.add(Activation('relu'))
model.add(Dense(200))
model.add(Activation('relu'))
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dense(1))

# compile, fit and save model
model.compile('adam', 'mse', ['mae'])
history = model.fit(X_train, y_train, epochs=10, validation_split=0.2, shuffle=True)
model.save('model.h5')