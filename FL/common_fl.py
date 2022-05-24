import keras
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.regularizers import l2


XY = Tuple[np.ndarray, np.ndarray]
XYList = List[XY]
PartitionedDataset = List[Tuple[XY, XY]]











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
    model = keras.Sequential()
    model.add(keras.Input(shape=(3875,)))
    model.add(keras.layers.Dense(units=2,kernel_initializer='glorot_uniform', activation='sigmoid'))
    return model




def load_data(idx: int):
        #print(str(idx))

        X_train = np.load("/home/sikha/data/X_train_"+str(idx)+".npy",allow_pickle=True)
        y_train = np.load("/home/sikha/data/y_train_"+str(idx)+".npy",allow_pickle=True)
        y_train_2 = []
        for label in y_train:
            if label == 1:
                x = [0,1]
            else:
                x = [1,0]
            y_train_2.append(np.array(x))
        y_train_2 = np.array(y_train_2)
        X_test = np.load("/home/sikha/data/X_test_"+str(idx)+".npy",allow_pickle=True)
        y_test = np.load("/home/sikha/data/y_test_"+str(idx)+".npy",allow_pickle=True)
        print(X_train.shape, X_test.shape)
        return (X_train, y_train_2), (X_test, y_test)
