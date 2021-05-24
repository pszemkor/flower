import argparse
import zipfile
from typing import Dict, Optional, Tuple
from google.cloud import storage

import flwr as fl
import tensorflow as tf
import wget
import pandas as pd
from client import load_test_data

BATCH_SIZE = 32
NUM_ROUNDS = 10
LOCAL_EPOCHS = 2
CLIENT_COUNT = 5
ROUND_NO = 0
STRATEGY = 'iid'

loss_vals = []
acc_vals = []


def upload_to_bucket(blob_name, path_to_file, bucket_name):
    storage_client = storage.Client.from_service_account_json('creds.json')

    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(path_to_file)

    return blob.public_url


def generate_csv():
    global ROUND_NO, BATCH_SIZE, LOCAL_EPOCHS, CLIENT_COUNT, loss_vals, acc_vals
    d = {'acc': acc_vals, 'loss': loss_vals}
    df = pd.DataFrame(d)
    filename = 'R_{}_LE_{}_CC_{}_BS_{}_STRATEGY_{}.csv'.format(str(ROUND_NO), str(LOCAL_EPOCHS),
                                                               str(CLIENT_COUNT), str(BATCH_SIZE), STRATEGY)
    df.to_csv(filename)
    upload_to_bucket(filename, filename, 'fl-covid-data')


def download_dataset():
    wget.download('https://storage.googleapis.com/fl-covid-data/test.zip')
    with zipfile.ZipFile('./test.zip', 'r') as zip_ref:
        zip_ref.extractall('.')


def main() -> None:
    # Load and compile model for
    # 1. server-side parameter initialization
    # 2. server-side parameter evaluation
    global BATCH_SIZE
    global NUM_ROUNDS
    global LOCAL_EPOCHS
    global CLIENT_COUNT
    global STRATEGY

    download_dataset()

    model = tf.keras.applications.EfficientNetB0(
        input_shape=(224, 224, 3), weights=None, classes=3
    )

    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--local_epochs", type=int, required=True)
    parser.add_argument("--rounds", type=int, required=True)
    parser.add_argument("--strategy", type=str, required=True)  # for logging purposes only
    parser.add_argument("--count", type=int, required=True)

    args = parser.parse_args()

    BATCH_SIZE = args.batch_size
    NUM_ROUNDS = args.rounds
    LOCAL_EPOCHS = args.local_epochs
    CLIENT_COUNT = args.count
    STRATEGY = args.strategy
    print("Strategy:", args.strategy)

    model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

    # Create strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.5,
        fraction_eval=0.5,
        min_fit_clients=CLIENT_COUNT,
        min_eval_clients=1,
        min_available_clients=CLIENT_COUNT,
        eval_fn=get_eval_fn(model),
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        initial_parameters=model.get_weights(),
    )

    # Start Flower server for four rounds of federated learning
    fl.server.start_server("[::]:8080", config={"num_rounds": NUM_ROUNDS}, strategy=strategy)


def get_eval_fn(model):
    """Return an evaluation function for server-side evaluation."""

    # Load data and model here to avoid the overhead of doing it in `evaluate` itself
    x_test, y_test = load_test_data()

    # The `evaluate` function will be called after every round
    def evaluate(
        weights: fl.common.Weights,
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        global ROUND_NO, loss_vals, acc_vals
        model.set_weights(weights)  # Update model with the latest parameters
        loss, accuracy = model.evaluate(x_test, y_test)
        acc_vals.append(accuracy)
        loss_vals.append(loss)
        print('ROUND', ROUND_NO, 'acc', accuracy, 'loss', loss)
        ROUND_NO += 1
        loss_vals.append(loss)
        acc_vals.append(accuracy)

        return loss, {"accuracy": accuracy}

    return evaluate


def fit_config(rnd: int):
    """Return training configuration dict for each round.
    """
    config = {
        "batch_size": BATCH_SIZE,
        "local_epochs": LOCAL_EPOCHS
    }
    return config


def evaluate_config(rnd: int):
    """Return evaluation configuration dict for each round.
    """
    val_steps = 10
    return {"val_steps": val_steps}


if __name__ == "__main__":
    main()
