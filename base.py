import os
import random
import cv2
import shutil
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras_tuner import RandomSearch
from sklearn import preprocessing
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, Rescaling
# from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam, RMSprop, Adagrad, SGD


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
    fig.savefig("img/random_sample.png")
    fig.show()

    le = preprocessing.LabelEncoder()
    labels = le.fit_transform(labels)
    data_array = np.asarray(data)
    labels_array = np.asarray(labels)

    return data_array, labels_array


def create_model(hp):
    model_architecture = Sequential()
    model_architecture.add(Rescaling(1. / 255, input_shape=(48, 48, 1)))
    model_architecture.add(Conv2D(32, (3, 3), padding="same", activation=hp.Choice("activation",
                                                                                   values=["relu", "tanh", "sigmoid"],
                                                                                   default="relu")))
    model_architecture.add(MaxPooling2D(pool_size=(2, 2)))
    model_architecture.add(Dropout(hp.Float("dropout",
                                            min_value=0.0,
                                            max_value=0.5,
                                            default=0.2,
                                            step=0.1)))

    model_architecture.add(Conv2D(64, (3, 3), padding="same", activation=hp.Choice("activation",
                                                                                   values=["relu", "tanh", "sigmoid"],
                                                                                   default="relu")))
    model_architecture.add(MaxPooling2D(pool_size=(2, 2)))
    model_architecture.add(Dropout(hp.Float("dropout",
                                            min_value=0.0,
                                            max_value=0.5,
                                            default=0.2,
                                            step=0.1)))

    model_architecture.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model_architecture.add(Dense(64, activation=hp.Choice("dense_activation",
                                                          values=["relu", "tanh", "sigmoid"],
                                                          default="relu")))
    model_architecture.add(Dropout(hp.Float("dropout",
                                            min_value=0.0,
                                            max_value=0.5,
                                            default=0.2,
                                            step=0.1)))

    model_architecture.add(Dense(7))
    model_architecture.add(Activation("softmax"))

    optimizer = hp.Choice("optimizer", values=["adam", "rmsprop", "sgd", "adagrad"], default="adam")

    # hp.Choice("momentum", values=[0.5, 0.9, 0.99], default=0.5)

    if optimizer == "adam":
        op = Adam(learning_rate=hp.Float("learning_rate", min_value=0.1, max_value=0.01, default=0.1, step=0.1))
    elif optimizer == "rmsprop":
        op = RMSprop(learning_rate=hp.Float("learning_rate", min_value=0.1, max_value=0.01, default=0.1, step=0.1))
    elif optimizer == "sgd":
        op = SGD(learning_rate=hp.Float("learning_rate", min_value=0.1, max_value=0.01, default=0.1, step=0.1))
    else:
        op = Adagrad(learning_rate=hp.Float("learning_rate", min_value=0.1, max_value=0.01, default=0.1, step=0.1))

    model_architecture.compile(loss="sparse_categorical_crossentropy", optimizer=op, metrics=["sparse_categorical_accuracy"])

    return model_architecture


def plot_loss_acc_from_history(hist, epoch_count):
    acc = hist.history["sparse_categorical_accuracy"]
    val_acc = hist.history["val_sparse_categorical_accuracy"]

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
    plt.savefig("img/random_sample.png")
    plt.show()


dataset = "ck-plus"
X, y = load_data(dataset)
X = np.expand_dims(X, 3)
w = X.shape[1]
h = X.shape[2]
print(f"Input shape: ({w}x{h})")
# total_emotions = len(Counter(y).keys())

shutil.rmtree("untitled_project/", ignore_errors=True)  # remove old results and forces a new run
tuner_rs = RandomSearch(
            create_model,
            objective="val_sparse_categorical_accuracy",
            seed=8,
            max_trials=10,
            executions_per_trial=2)

n_epochs = 25
with tf.device("/gpu:0"):
    print("Starting tuner...")
    tuner_rs.search(X, y, epochs=n_epochs, validation_split=0.2, verbose=1)

print("Retraining best model...")
best_model = tuner_rs.get_best_models()[0]
history = best_model.fit(X, y, epochs=n_epochs, validation_split=0.2, verbose=1)
plot_loss_acc_from_history(history, n_epochs)
best_model.save(f"models/{dataset}.h5")
