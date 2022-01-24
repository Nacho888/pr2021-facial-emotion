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

            if dataset_name == "jaffe":
                image = image[80:230, 50:200]
                if width is not None and height is not None:
                    image = cv2.resize(image, (width, height))
                img_flip_lr = cv2.flip(image, 1)
                img_rotate_90_clockwise = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                img_rotate_90_counter_clockwise = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                img_rotate_180 = cv2.rotate(image, cv2.ROTATE_180)

                images = [image, img_flip_lr, img_rotate_90_clockwise, img_rotate_90_counter_clockwise, img_rotate_180]

                for img in images:
                    data.append(img)
                    labels.append(emotion)
            else:
                if width is not None and height is not None:
                    image = cv2.resize(image, (width, height))

                data.append(image)
                labels.append(emotion)

    plt.hist(labels, bins=np.arange(7) - 0.5, rwidth=0.8)  # set arange to number of classes + 1
    plt.title(f"'{dataset_name}' Class distribution (N={len(labels)})")
    plt.xlabel("Class label")
    plt.ylabel("Frequency")
    plt.savefig(f"img/{dataset_name}_class_distribution.png")
    plt.show()

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

    model_architecture.add(Dense(6))
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


# MAIN
datasets = [{"name": "jaffe", "raw_data": [], "matrices": [], "model": None, "scores": []},
            {"name": "ck-plus", "raw_data": [], "matrices": [], "model": None, "scores": []}]

for dataset in datasets:
    X, y = load_data(dataset["name"], 48, 48)
    X = np.expand_dims(X, 3)
    w = X.shape[1]
    h = X.shape[2]
    print(f"Input shape: ({w}x{h})")

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
        stop_early = EarlyStopping(monitor="val_loss", min_delta=0.075, patience=3)
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
    eval_result = hypermodel.evaluate(X_test, y_test, verbose=0)
    print(f"{dataset['name']} dataset [test loss, test accuracy]:", eval_result)
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, hypermodel.predict_classes(X_test)))
    disp.plot(values_format="d")
    disp.ax_.set_title(f"{dataset['name']} model with non-mixed test data")
    plt.savefig(f"img/{dataset['name']}_confusion_matrix.png")
    plt.show()

    # Plot loss vs accuracy
    plot_loss_acc_from_history(history, dataset['name'])

    dataset["raw_data"] = [X, y]
    dataset["matrices"] = [X_train, X_test, y_train, y_test]
    dataset["model"] = hypermodel
    dataset["scores"] = [hypermodel.evaluate(X_train, y_train, verbose=0), eval_result]

# JAFFE
eval_result = datasets[0]["model"].evaluate(datasets[1]["matrices"][1], datasets[1]["matrices"][3], verbose=0)
print("Asian model using Caucasian data [test loss, test accuracy]:", eval_result)
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(datasets[1]["matrices"][3],
                                                                datasets[0]["model"].predict_classes(datasets[1]["matrices"][1])))
disp.plot(values_format="d")
disp.ax_.set_title(f"{datasets[0]['name']} model with Caucasian test data")
plt.savefig(f"img/{datasets[0]['name']}_caucasian_confusion_matrix.png")
plt.show()

# CK+
eval_result = datasets[1]["model"].evaluate(datasets[0]["matrices"][1], datasets[0]["matrices"][3], verbose=0)
print("Caucasian model using Asian data [test loss, test accuracy]:", eval_result)
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(datasets[0]["matrices"][3],
                                                                datasets[1]["model"].predict_classes(datasets[0]["matrices"][1])))
disp.plot(values_format="d")
disp.ax_.set_title(f"{datasets[1]['name']} model with Asian test data")
plt.savefig(f"img/{datasets[1]['name']}_asian_confusion_matrix.png")
plt.show()

# Summary of results
for dataset in datasets:
    print(f"{dataset['name']} dataset:")
    print(f"\tTraining score: {dataset['scores'][0][1]}")
    print(f"\tTest score: {dataset['scores'][1][1]}")
