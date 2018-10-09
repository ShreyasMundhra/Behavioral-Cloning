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

# def create_dataset():
#     data_path = '/opt/carnd_p3/data/'
#     df = pd.read_csv(data_path + 'driving_log.csv')
#     df['steering'] = pd.to_numeric(df['steering'])
    
#     center_df = df[['center', 'steering']]
    
#     correction = 0.2
    
#     left_df = df[['left', 'steering']]
#     left_df.rename(columns={'left': 'center'}, inplace=True)
#     left_df['steering'] = left_df['steering'] + correction
    
#     right_df = df[['right', 'steering']]
#     right_df.rename(columns={'right': 'center'}, inplace=True)
#     right_df['steering'] = right_df['steering'] - correction
    
#     translated_df = pd.concat([center_df, left_df, right_df], ignore_index=True)    
#     lines = translated_df.values
#     return lines

# def generator(lines, batch_size=32):
#     data_path = '/opt/carnd_p3/data/'
#     num_lines = len(lines)
    
#     while 1:
#         lines = shuffle(lines)
#         for offset in range(0, num_lines, batch_size):
#             batch_samples = lines[offset: offset + batch_size]
            
#             images = []
#             measurements = []

#             for batch_sample in batch_samples:
#                 image = ndimage.imread(data_path + 'IMG/' + batch_sample[0].split('/')[-1])
#                 flipped_image = np.fliplr(image)

#                 steering = float(batch_sample[1])
#                 flipped_steering = -steering
                
#                 index = np.random.randint(0, 2)
#                 sample_img = [image, flipped_image][index]
#                 sample_steering = [steering, flipped_steering][index]

#                 images.append(sample_img)
#                 measurements.append(sample_steering)

#             X_train = np.array(images)
#             y_train = np.array(measurements)
# #             yield shuffle(X_train, y_train)
#             yield X_train, y_train

def load_data():
    data_path = '/opt/carnd_p3/data/'
    lines = []
    
    df = pd.read_csv(data_path + 'driving_log.csv')
    lines = list(df.values)
#     with open(data_path + 'driving_log.csv') as csvfile:
#         reader = csv.reader(csvfile)
#         for line in reader:
#             lines.append(line)
            
    images = []
    measurements = []
    for i in range(1, len(lines)):
        line = lines[i]
        
        image_center = ndimage.imread(data_path + 'IMG/' + line[0].split('/')[-1])
        image_left = ndimage.imread(data_path + 'IMG/' + line[1].split('/')[-1])
        image_right = ndimage.imread(data_path + 'IMG/' + line[2].split('/')[-1])
        
        flipped_image_center = np.fliplr(image_center)
        flipped_image_left = np.fliplr(image_left)
        flipped_image_right = np.fliplr(image_right)
        
        steering_center = float(line[3])
        correction = 0.2 # this is a parameter to tune
        steering_left = steering_center + correction
        steering_right = steering_center - correction
        
        flipped_steering_center = -steering_center
        flipped_steering_left = -steering_left
        flipped_steering_right = -steering_right
        
        images.extend([image_center, image_left, image_right, flipped_image_center, flipped_image_left, flipped_image_right])
        measurements.extend([steering_center, steering_left, steering_right, flipped_steering_center, flipped_steering_left, flipped_steering_right])
    
    X_train = np.array(images)
    y_train = np.array(measurements)
    return X_train, y_train

def grayscale(img):
    import tensorflow as tf
    return tf.image.rgb_to_grayscale(img)

# samples = create_dataset()
# train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# # compile and train the model using the generator function
# train_generator = generator(train_samples, batch_size=32)
# validation_generator = generator(validation_samples, batch_size=32)

X_train, y_train = load_data()

model = Sequential()
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
# model.add(Lambda(lambda x: grayscale(x)))
# print(model.layers[0].get_output_at(0).get_shape().as_list())
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
# print(model.layers[7].get_output_at(0).get_shape().as_list())
model.add(Dense(5000))
model.add(Activation('relu'))
model.add(Dense(1000))
model.add(Activation('relu'))
model.add(Dense(200))
model.add(Activation('relu'))
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dense(1))

# model = load_model('model.h5')

# compile and fit model
model.compile('adam', 'mse', ['mae'])
history = model.fit(X_train, y_train, epochs=10, validation_split=0.2, shuffle=True)
# model.fit_generator(train_generator, samples_per_epoch=
#                     len(train_samples), validation_data=validation_generator,
#                     nb_val_samples=len(validation_samples), nb_epoch=3)

model.save('model.h5')