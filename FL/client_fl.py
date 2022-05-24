import argparse
import os

import tensorflow as tf

import flwr as fl
import numpy as np
import common
RANDOM_SEED = 1000
#seed(47568)
#tf.random.set_random_seed(seed_value)
os.environ['PYTHONHASHSEED']=str(RANDOM_SEED)
#random.seed(47568)
tf.random.set_seed(RANDOM_SEED)

#SAVE = False

np.random.seed(RANDOM_SEED)

# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"



# Define Flower client
class MnistClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_test, y_test, args):
        self.model = model
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test
        self.batch_size = args.batch_size
        self.local_epochs = args.local_epochs

        model.compile("sgd", loss="binary_crossentropy", metrics=[tf.keras.metrics.Recall()])

    def get_parameters(self):
        """Get parameters of the local model."""
        raise Exception("Not implemented (server-side parameter initialization)")

    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""
        # Update local model parameters

        self.model.set_weights(parameters)
        # Train the model
        self.model.fit(
            self.x_train,
            self.y_train,
            epochs=self.local_epochs,
            batch_size=self.batch_size,
        )

        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""

        # Update local model with global parameters
        self.model.set_weights(parameters)

        # Evaluate global model parameters on the local test data and return results
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test)
        num_examples_test = len(self.x_test)
        return loss, num_examples_test, {"accuracy": accuracy}


def main(args) -> None:
    # Load Keras model
    model = common.create_mlp_model()

    # Load a subset of MNIST to simulate the local data partition
    (x_train, y_train), (x_test, y_test) = common.load_data(args.partition)



    # Start Flower client
    client = MnistClient(model, x_train, y_train, x_test, y_test, args)
    fl.client.start_numpy_client("[::]:8080", client=client)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flower Client")

    parser.add_argument("--partition", default=92, type=int, required=False)
    parser.add_argument(
        "--local-epochs",
        default=2,
        type=int,
        help="Total number of local epochs to train",
    )
    parser.add_argument("--batch-size", default=24, type=int, help="Batch size")
    parser.add_argument(
        "--learning-rate", default=0.1, type=float, help="Learning rate for training"
    )

    args = parser.parse_args()

    main(args)

