import os
import random
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras_tuner import RandomSearch
from sklearn import preprocessing
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, Rescaling
from tensorflow.keras.callbacks import EarlyStopping
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
    fig.savefig(f"img/{dataset_name}_random_sample.png")
    fig.show()

    le = preprocessing.LabelEncoder()
    labels = le.fit_transform(labels)
    data_array = np.asarray(data)
    labels_array = np.asarray(labels)

    return data_array, labels_array


def create_model(hp):
    model_architecture = Sequential()
    model_architecture.add(Rescaling(1. / 255, input_shape=(48, 48, 1)))
    model_architecture.add(Conv2D(32, (3, 3), padding="same", activation=hp.Choice("activation1",
                                                                                   values=["relu", "tanh", "sigmoid"],
                                                                                   default="relu")))
    model_architecture.add(MaxPooling2D(pool_size=(2, 2)))
    model_architecture.add(Dropout(hp.Float("dropout1",
                                            min_value=0.0,
                                            max_value=0.5,
                                            default=0.2,
                                            step=0.1)))

    model_architecture.add(Conv2D(64, (3, 3), padding="same", activation=hp.Choice("activation2",
                                                                                   values=["relu", "tanh", "sigmoid"],
                                                                                   default="relu")))
    model_architecture.add(MaxPooling2D(pool_size=(2, 2)))
    model_architecture.add(Dropout(hp.Float("dropout2",
                                            min_value=0.0,
                                            max_value=0.5,
                                            default=0.2,
                                            step=0.1)))

    model_architecture.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model_architecture.add(Dense(64, activation=hp.Choice("dense_activation",
                                                          values=["relu", "tanh", "sigmoid"],
                                                          default="relu")))
    model_architecture.add(Dropout(hp.Float("dropout3",
                                            min_value=0.0,
                                            max_value=0.5,
                                            default=0.2,
                                            step=0.1)))

    model_architecture.add(Dense(7))
    model_architecture.add(Activation("softmax"))

    optimizer = hp.Choice("optimizer", values=["adam", "rmsprop", "sgd", "adagrad"], default="adam")

    # hp.Choice("momentum", values=[0.5, 0.9, 0.99], default=0.5)

    with hp.conditional_scope("optimizer", optimizer):
        if optimizer == "adam":
            model_architecture.compile(optimizer=Adam(hp.Float("learning_rate",
                                                               min_value=0.0001,
                                                               max_value=0.01,
                                                               default=0.001)),
                                       loss="sparse_categorical_crossentropy",
                                       metrics=["sparse_categorical_accuracy"])
        elif optimizer == "rmsprop":
            model_architecture.compile(optimizer=RMSprop(hp.Float("learning_rate",
                                                                  min_value=0.0001,
                                                                  max_value=0.01,
                                                                  default=0.001)),
                                       loss="sparse_categorical_crossentropy",
                                       metrics=["sparse_categorical_accuracy"])
        elif optimizer == "sgd":
            model_architecture.compile(optimizer=SGD(hp.Float("learning_rate",
                                                              min_value=0.0001,
                                                              max_value=0.01,
                                                              default=0.001)),
                                       loss="sparse_categorical_crossentropy",
                                       metrics=["sparse_categorical_accuracy"])
        elif optimizer == "adagrad":
            model_architecture.compile(optimizer=Adagrad(hp.Float("learning_rate",
                                                                  min_value=0.0001,
                                                                  max_value=0.01,
                                                                  default=0.001)),
                                       loss="sparse_categorical_crossentropy",
                                       metrics=["sparse_categorical_accuracy"])

    return model_architecture


def plot_loss_acc_from_history(hist, dataset_name):
    acc = hist.history["sparse_categorical_accuracy"]
    val_acc = hist.history["val_sparse_categorical_accuracy"]

    loss = hist.history["loss"]
    val_loss = hist.history["val_loss"]

    epochs_range = range(len(hist.history["loss"]))

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Training Accuracy")
    plt.plot(epochs_range, val_acc, label="Validation Accuracy")
    plt.legend(loc="lower right")
    plt.title("Training and Validation Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Training Loss")
    plt.plot(epochs_range, val_loss, label="Validation Loss")
    plt.legend(loc="upper right")
    plt.title("Training and Validation Loss")
    plt.savefig(f"img/{dataset_name}_loss_vs_acc.png")
    plt.show()


dataset = "ck-plus"
X, y = load_data(dataset)
X = np.expand_dims(X, 3)
w = X.shape[1]
h = X.shape[2]
print(f"Input shape: ({w}x{h})")
# total_emotions = len(Counter(y).keys())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=8)

tuner_rs = RandomSearch(
    create_model,
    objective="val_sparse_categorical_accuracy",
    seed=8,
    max_trials=10,
    executions_per_trial=2,
    overwrite=True)

n_epochs = 15
with tf.device("/gpu:0"):  # if there is an available GPU, use it
    print("Starting tuner...")
    stop_early = EarlyStopping(monitor="val_loss", patience=5)
    tuner_rs.search(X_train, y_train, epochs=n_epochs, validation_split=0.1, verbose=1, callbacks=[stop_early])

# Retrain best model on full dataset with best hyperparameters
print("Retraining best model...")
best_hps = tuner_rs.get_best_hyperparameters(num_trials=1)[0]
hypermodel = tuner_rs.hypermodel.build(best_hps)
history = hypermodel.fit(X_train, y_train, epochs=50, validation_split=0.1, callbacks=[stop_early])
print("Best model architecture")
print(hypermodel.summary())
print("Best model hyperparams")
tuner_rs.results_summary(1)

# Evaluate and plot confusion matrix
eval_result = hypermodel.evaluate(X_test, y_test)
print("[test loss, test accuracy]:", eval_result)
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, hypermodel.predict_classes(X_test)))
disp.plot(values_format="d")
plt.savefig(f"img/{dataset}_confusion_matrix.png")
plt.show()

# Plot loss vs accuracy
plot_loss_acc_from_history(history, dataset)
