import argparse
import os
import zipfile

import cv2
import flwr as fl

import numpy as np
import sklearn
import tensorflow as tf
import wget
from tensorflow.keras.utils import to_categorical

# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

class_encoding = {'COVID': 0, 'Normal': 1, 'Viral Pneumonia': 2}


def download_dataset():
    wget.download('https://storage.googleapis.com/fl-covid-data/train_valid.zip')
    with zipfile.ZipFile('./train_valid.zip', 'r') as zip_ref:
        zip_ref.extractall('.')


def get_samples_from(dir_path):
    print('getting samples from', dir_path)
    samples = []
    for sample in os.listdir(dir_path):
        samples.append(os.path.join(dir_path, sample))
    return samples


def get_all_vals(class_dict):
    x = []
    y = []
    for k, v in class_dict.items():
        for sample in v:
            x.append(sample)
            y.append(k)

    y = np.array(list(map(lambda label: class_encoding[label], y)))
    return np.array(x), y


def get_iid(id, class_to_samples, count):
    x, y = get_all_vals(class_to_samples)
    return transform_to_train(id, x, y, count)


def get_random(id, class_to_samples, count):
    x, y = get_all_vals(class_to_samples)
    sklearn.utils.shuffle(x, y)
    return transform_to_train(id, x, y, count)


def get_noniid(id, class_to_samples, count):
    x, y = get_all_vals(class_to_samples)
    n = len(x)
    batch_size = n // count
    x_train = np.array(map(lambda path: load_image(path), x[batch_size * id: batch_size * (id + 1)]))
    y_train = np.array(y[batch_size * id: batch_size * (id + 1)])
    return x_train, y_train


def transform_to_train(id, x, y, count):
    x_train, y_train = [], []
    i = id
    while i < len(x):
        img = load_image(x[i])
        x_train.append(img)
        y_train.append(y[i])
        i += count
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train, y_train = sklearn.utils.shuffle(x_train, y_train, random_state=42)
    return x_train, y_train


def load_train_data(id, strategy, count):
    class_to_train_samples = read_data_from_path('train')
    print(class_to_train_samples.keys())
    for k, v in class_to_train_samples.items():
        print(k, v[0], v[-1])
        print("****")
    print(len(class_to_train_samples))

    if strategy == 'iid':
        return get_iid(id, class_to_train_samples, count)
    elif strategy == 'noniid':
        return get_noniid(id, class_to_train_samples, count)
    else:
        return get_random(id, class_to_train_samples, count)


def load_valid_data(id, count):
    class_to_valid = read_data_from_path('valid')
    return get_iid(id, class_to_valid, count)


def load_test_data():
    class_to_samples = read_data_from_path('test')
    x, y = get_all_vals(class_to_samples)
    x_test, y_test = [], []
    for i in range(len(x)):
        img = load_image(x[i])
        x_test.append(img)
        y_test.append(y[i])
    x_test, y_test = np.array(x_test), np.array(y_test)
    x_test, y_test = sklearn.utils.shuffle(x_test, y_test, random_state=42)

    return x_test, y_test


def read_data_from_path(path):
    class_to_train_samples = {}

    for item in os.listdir(path):
        if item in class_encoding.keys():
            class_to_train_samples[item] = get_samples_from(os.path.join(path, item))[:100] #todo remove - just to speed up debugging on GCP
    return class_to_train_samples


def load_image(path):
    return cv2.imread(path, cv2.IMREAD_COLOR)


# Define Flower client
class FederatedClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test

    def get_parameters(self):
        """Get parameters of the local model."""
        raise Exception("Not implemented (server-side parameter initialization)")

    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""

        # Update local model parameters
        self.model.set_weights(parameters)

        # Get hyperparameters for this round
        batch_size: int = config["batch_size"]
        epochs: int = config["local_epochs"]

        # Train the model using hyperparameters from config
        history = self.model.fit(
            self.x_train,
            self.y_train,
            batch_size,
            epochs,
            validation_split=0.2,
        )

        # Return updated model parameters and results
        parameters_prime = self.model.get_weights()
        num_examples_train = len(self.x_train)
        results = {
            "loss": history.history["loss"][0],
            "accuracy": history.history["accuracy"][0],
            "val_loss": history.history["val_loss"][0],
            "val_accuracy": history.history["val_accuracy"][0],
        }
        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""

        # Update local model with global parameters
        self.model.set_weights(parameters)

        # Get config values
        steps: int = config["val_steps"]

        # Evaluate global model parameters on the local test data and return results
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, 32)
        num_examples_test = len(self.x_test)
        return loss, num_examples_test, {"accuracy": accuracy}


def main() -> None:
    # Parse command line argument `partition`
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument("--partition", type=int, required=True)
    parser.add_argument("--server", type=str, required=True)
    parser.add_argument("--strategy", type=str, required=True)
    parser.add_argument("--count", type=int, required=True)

    args = parser.parse_args()

    # Load and compile Keras model
    model = tf.keras.applications.EfficientNetB0(
        input_shape=(224, 224, 3), weights=None, classes=3
    )
    model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    download_dataset()
    x_train, y_train = load_train_data(args.partition, args.strategy, args.count)
    print(len(x_train), len(y_train))
    x_test, y_test = load_valid_data(args.partition, args.count)
    print(len(x_test), len(y_test))

    # Start Flower client
    client = FederatedClient(model, x_train, y_train, x_test, y_test)
    fl.client.start_numpy_client(args.server + ":8080", client=client)


if __name__ == "__main__":
    main()
