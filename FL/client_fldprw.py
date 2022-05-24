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

from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras_vectorized import (
    VectorizedDPKerasSGDOptimizer
)
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasSGDOptimizer

PRIVACY_LOSS = 0

# Define Flower client
class MnistClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_test, y_test, args):
        self.model = model
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test
        self.batch_size = args.batch_size
        self.local_epochs = args.local_epochs
        self.dpsgd = args.dpsgd
        if args.dpsgd:
            self.noise_multiplier = args.noise_multiplier
            if args.batch_size % args.microbatches != 0:
                raise ValueError(
                    "Number of microbatches should divide evenly batch_size"
                )
            optimizer = VectorizedDPKerasSGDOptimizer(
                l2_norm_clip=args.l2_norm_clip,
                noise_multiplier=args.noise_multiplier,
                num_microbatches=args.microbatches,
                learning_rate=args.learning_rate,
            )
            #opt = DPKerasSGD(l2_norm_clip=1.0, noise_multiplier=0.5, num_microbatches=1,
            #      <standard arguments>)
            # Compute vector of per-example loss rather than its mean over a minibatch.
            loss = tf.keras.losses.BinaryCrossentropy(from_logits=True,reduction=tf.losses.Reduction.NONE)
        else:
            optimizer = tf.keras.optimizers.SGD(learning_rate=args.learning_rate)
            loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        # Compile model with Keras #
        model.compile(optimizer=optimizer, loss=loss, metrics=[tf.keras.metrics.Recall()])
        

    def get_parameters(self):
        """Get parameters of the local model."""
        raise Exception("Not implemented (server-side parameter initialization)")

    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""
        # Update local model parameters

        global PRIVACY_LOSS
        if self.dpsgd:
            privacy_spent = common.compute_epsilon(
                self.local_epochs,
                len(self.x_train),
                self.batch_size,
                self.noise_multiplier,
            )
            PRIVACY_LOSS += privacy_spent

        self.model.set_weights(parameters)
        unprot_class_weight_0: float = config['0UP'] #male 0
        unprot_class_weight_1: float = config['1UP'] #male 1
        prot_class_weight_0: float = config['0P'] #female 0
        prot_class_weight_1: float = config['1P'] #female 1

        sample_weights = []
        is_prot_array = self.x_train[:,1]
        y_train = self.y_train[:,1]
        for label,prot_value in zip(y_train,is_prot_array):
                if label == 0 and prot_value == 0:  #male with 0
                    sample_weights.append(unprot_class_weight_0)
                elif label == 0 and prot_value == 1:  #female with 0
                    sample_weights.append(prot_class_weight_0)
                elif label == 1 and prot_value == 1:  #female with 0
                    sample_weights.append(prot_class_weight_1)
                else:
                    sample_weights.append(unprot_class_weight_1)
        sample_weights = np.array(sample_weights)
        # Train the model
        self.model.fit(
            self.x_train,
            self.y_train,
            epochs=self.local_epochs,
            batch_size=self.batch_size,
            sample_weight=sample_weights
        )
        print("=======================================================================================>", config["rnd"])

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

    if args.dpsgd and x_train.shape[0] % args.batch_size != 0:
        drop_num = x_train.shape[0] % args.batch_size
        x_train = x_train[:-drop_num]
        y_train = y_train[:-drop_num]

    # Start Flower client
    client = MnistClient(model, x_train, y_train, x_test, y_test, args)
    fl.client.start_numpy_client("[::]:8080", client=client)
    if args.dpsgd:
        print("Privacy Loss: ", PRIVACY_LOSS)


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
    parser.add_argument(
        "--dpsgd",
        default=True,
        type=bool,
        help="If True, train with DP-SGD. If False, " "train with vanilla SGD.",
    )
    parser.add_argument("--l2-norm-clip", default=1.0, type=float, help="Clipping norm")
    parser.add_argument(
        "--noise-multiplier",
        default=1.1,
        type=float,
        help="Ratio of the standard deviation to the clipping norm",
    )
    parser.add_argument(
        "--microbatches",
        default=24,
        type=int,
        help="Number of microbatches " "(must evenly divide batch_size)",
    )
    args = parser.parse_args()

    main(args)

