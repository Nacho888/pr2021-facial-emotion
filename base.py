import os
import random
import cv2
import numpy as np
from collections import Counter
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, Rescaling


def load_data(dataset_name, width=None, height=None):
    data = []
    labels = []
    for emotion in os.listdir(f"datasets/{dataset_name}"):
        for image_name in os.listdir(f"datasets/{dataset_name}/{emotion}"):
            image = cv2.imread(f"datasets/{dataset_name}/{emotion}/{image_name}", cv2.IMREAD_GRAYSCALE)
            if width is not None and height is not None:
                image = cv2.resize(image, (width, height))
            data.append(image)
            labels.append(emotion)

    fig, axs = plt.subplots(3, 3)
    for _i in range(len(axs)):
        for _j in range(len(axs)):
            index = random.randint(0, len(data))
            axs[_i, _j].imshow(data[index], cmap=plt.cm.gray)
            axs[_i, _j].set_title(labels[index])
    fig.tight_layout(pad=1.0)
    # fig.savefig("../img/random_sample.png")
    fig.show()

    le = preprocessing.LabelEncoder()
    labels = le.fit_transform(labels)
    data_array = np.asarray(data)
    labels_array = np.asarray(labels)

    return data_array, labels_array


def create_model(class_count, width, height):
    model_architecture = Sequential()
    model_architecture.add(Rescaling(1./255, input_shape=(width, height, 1)))
    model_architecture.add(Conv2D(32, (3, 3), ))
    model_architecture.add(Activation("relu"))
    model_architecture.add(MaxPooling2D(pool_size=(2, 2)))
    model_architecture.add(Dropout(0.25))

    model_architecture.add(Conv2D(64, (3, 3)))
    model_architecture.add(Activation("relu"))
    model_architecture.add(MaxPooling2D(pool_size=(2, 2)))
    model_architecture.add(Dropout(0.25))

    model_architecture.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model_architecture.add(Dense(64))
    model_architecture.add(Activation("relu"))
    model_architecture.add(Dropout(0.25))
    model_architecture.add(Dense(class_count))
    model_architecture.add(Activation("softmax"))

    model_architecture.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model_architecture


def plot_loss_acc_from_history(hist, epoch_count):
    acc = hist.history["accuracy"]
    val_acc = hist.history["val_accuracy"]

    loss = hist.history["loss"]
    val_loss = hist.history["val_loss"]

    epochs_range = range(epoch_count)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    # plt.savefig("../img/random_sample.png")
    plt.show()


dataset = "ck-plus"
X, y = load_data(dataset)
w = X.shape[1]
h = X.shape[2]
print(w, h)
total_emotions = len(Counter(y).keys())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train = np.expand_dims(X_train, 3)
X_test = np.expand_dims(X_test, 3)

model = create_model(total_emotions, w, h)
model.summary()
n_epochs = 25
history = model.fit(X_train, y_train, batch_size=32, epochs=n_epochs, validation_data=(X_test, y_test))
plot_loss_acc_from_history(history, n_epochs)
# model.save(f"models/{dataset}.h5")
