from typing import List, Tuple

import numpy as np
import tensorflow as tf

from tensorflow_privacy.privacy.analysis.rdp_accountant import compute_rdp
from tensorflow_privacy.privacy.analysis.rdp_accountant import get_privacy_spent
from tensorflow.keras.regularizers import l2
from tensorflow.keras.regularizers import L2

XY = Tuple[np.ndarray, np.ndarray]
XYList = List[XY]
PartitionedDataset = List[Tuple[XY, XY]]


def compute_epsilon(
    epochs: int, num_train_examples: int, batch_size: int, noise_multiplier: float
) -> float:
    """Computes epsilon value for given hyperparameters.

    Based on
    github.com/tensorflow/privacy/blob/master/tutorials/mnist_dpsgd_tutorial_keras.py
    """
    if noise_multiplier == 0.0:
        return float("inf")
    steps = epochs * num_train_examples // batch_size
    orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
    sampling_probability = batch_size / num_train_examples
    rdp = compute_rdp(
        q=sampling_probability,
        noise_multiplier=noise_multiplier,
        steps=steps,
        orders=orders,
    )
    # Delta is set to approximate 1 / (number of training points).
    return get_privacy_spent(orders, rdp, target_delta=1 / num_train_examples)[0]


def create_mlp_model() -> tf.keras.Model:
    """Returns a sequential keras CNN Model."""
    '''
    return tf.keras.Sequential(
        [
    
    
            tf.keras.layers.Dense(1,
                                  activation = 'sigmoid',
                                  kernel_regularizer=L2(l2=1),
                                  input_dim = 3875)
        ]
    )
    '''
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(units=1,kernel_initializer='glorot_uniform', activation='sigmoid',kernel_regularizer=l2(0.),
                                    input_dim = 3875))
    return model



def shuffle(X: np.ndarray, y: np.ndarray) -> XY:
    """Shuffle X and y."""
    rng = np.random.default_rng()
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def partition(X: np.ndarray, y: np.ndarray, num_partitions: int) -> XYList:
    """Split X and y into a number of partitions."""
    return list(
        zip(np.array_split(X, num_partitions), np.array_split(y, num_partitions))
    )


def preprocess(X: np.ndarray, y: np.ndarray) -> XY:
    """Basic preprocessing for MNIST dataset."""
    X = np.array(X, dtype=np.float32) / 255
    X = X.reshape((X.shape[0], 28, 28, 1))

    y = np.array(y, dtype=np.int32)
    y = tf.keras.utils.to_categorical(y, num_classes=10)

    return X, y


def create_partitions(source_dataset: XY, num_partitions: int) -> XYList:
    """Create partitioned version of a source dataset."""
    X, y = source_dataset
    X, y = shuffle(X, y)
    X, y = preprocess(X, y)
    xy_partitions = partition(X, y, num_partitions)
    return xy_partitions


def load(
    num_partitions: int,
) -> PartitionedDataset:
    """Create partitioned version of MNIST."""
    xy_train, xy_test = tf.keras.datasets.mnist.load_data()
    xy_train_partitions = create_partitions(xy_train, num_partitions)
    xy_test_partitions = create_partitions(xy_test, num_partitions)
    return list(zip(xy_train_partitions, xy_test_partitions))

def load_data(idx: int):
        #print(str(idx))

        X_train = np.load("/Users/sikha/0FAIRFL/Data/113p/X_train_"+str(idx)+".npy",allow_pickle=True)
        y_train = np.load("/Users/sikha/0FAIRFL/Data/113p/y_train_"+str(idx)+".npy",allow_pickle=True)
        y_train_2 = []
        for label in y_train:
            if label == 1:
                x = [0,1]
            else:
                x = [1,0]
            y_train_2.append(np.array(x))
        y_train_2 = np.array(y_train_2)
        X_test = np.load("/Users/sikha/0FAIRFL/Data/113p/X_test_"+str(idx)+".npy",allow_pickle=True)
        y_test = np.load("/Users/sikha/0FAIRFL/Data/113p/y_test_"+str(idx)+".npy",allow_pickle=True)
        return (X_train, y_train_2), (X_test, y_test)
