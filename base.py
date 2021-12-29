import os
import cv2
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, \
    BatchNormalization, Reshape


def load_data(dataset_name, width, height):
    data = []
    labels = []
    for emotion in os.listdir(f"datasets/{dataset_name}"):
        for image_name in os.listdir(f"datasets/{dataset_name}/{emotion}"):
            image = cv2.imread(f"datasets/{dataset_name}/{emotion}/{image_name}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert to grayscale
            image = cv2.resize(image, (width, height))  # resize image to 48x48
            data.append(np.asarray(image))
            labels.append(emotion)

    data = np.asarray(data)
    labels = np.asarray(labels)

    return data, labels


def create_model():
    model = Sequential()
    model.add(Reshape((48, 48, 1), input_shape=(48, 48)))
    model.add(Conv2D(32, (3, 3), input_shape=(48, 48, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


dataset = "ck-plus"
width = 48
height = 48
input_shape = (width, height, 1)
X, y = load_data(dataset, width, height)
total_emotions = len(Counter(y).keys())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = create_model()
model.summary()
model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test))
model.save(f"models/{dataset}.h5")
