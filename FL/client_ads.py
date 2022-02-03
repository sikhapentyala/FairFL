import warnings
import flwr as fl
import numpy as np
#from flwr.common.typing import Config
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss,roc_auc_score
import os
import argparse
from sklearn.model_selection import train_test_split

RANDOM_SEED = 47568#int(time.time())

np.random.seed(RANDOM_SEED)
import utils_ads as utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument("--partition", type=int, choices=range(1,121), required=True)
    args = parser.parse_args()

    #ignore_users = [91,25,52,106,57,104,69]
    #if args.partition in ignore_users:
    #    pass
    def load_data(idx: int):
        #print(str(idx))

        X_train = np.load("/Users/sikha/0FAIRFL/Data/113p/X_train_"+str(idx)+".npy",allow_pickle=True)
        y_train = np.load("/Users/sikha/0FAIRFL/Data/113p/y_train_"+str(idx)+".npy",allow_pickle=True)
        X_test = np.load("/Users/sikha/0FAIRFL/Data/113p/X_test_"+str(idx)+".npy",allow_pickle=True)
        y_test = np.load("/Users/sikha/0FAIRFL/Data/113p/y_test_"+str(idx)+".npy",allow_pickle=True)
        return (X_train, y_train), (X_test, y_test)

    def add_laplacian_noise(count: int, epsilon: float):
        myscale = 1/epsilon
        noise = np.random.laplace(0., myscale, 1)
        return noise

    # Split train set into 10 partitions and randomly use one for training.
    (X_train, y_train),(X_test, y_test)  = load_data(args.partition)

    # Get count of y+ and y- in train
    count_pos = sum(y_train)
    count_neg = len(y_train) - count_pos

    # Add noise to each variable
    count_pos = count_pos + add_laplacian_noise(count_pos,epsilon=0.5)
    count_neg = count_neg + add_laplacian_noise(count_neg,epsilon=0.5)

    pre_proc_counts = {'1':count_pos, '0':count_neg}
    print("Client noise ",args.partition, pre_proc_counts)


    # Create LogisticRegression Model

    model = LogisticRegression(
        random_state = RANDOM_SEED, solver='sag',
        penalty="l2",#class_weight=pre_proc_counts,#'balanced',
        max_iter=2,  # local epoch
        warm_start=True,  # prevent refreshing weights when fitting
    )

    # Setting initial parameters, akin to model.compile for keras models
    utils.set_initial_params(model)
    #utils.set_initial_params(model)

    # Define Flower client
    class MnistClient(fl.client.NumPyClient):
        def get_parameters(self):  # type: ignore
            return utils.get_model_parameters(model)

        # send it to server

        #request_properties: fl.common.Config = {"1": "float", "0": "float"}
        #ins: fl.common.PropertiesIns = fl.common.PropertiesIns(
        #    config=request_properties
        #)
        #def get_properties(self, ins):
            # value: flwr.common.PropertiesRes =
        #    return utils.get_data_counts(pre_proc_counts)

        def fit(self, parameters, config):  # type: ignore
            utils.set_model_params(model, parameters)
            # Receive global DP-count from server
            class_weight_0: float = config['0']
            class_weight_1: float = config['1']
            sample_weights = []
            for label in y_train:
                if label == 0:
                    sample_weights.append(class_weight_0)
                else:
                    sample_weights.append(class_weight_1)
            # Ignore convergence failure due to low local epochs
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                #print(len(np.unique(y_train)))
                #if len(np.unique(y_train)) < 2:
                    #train with single class
                #    value_of_intercept = np.unique(y_train)[0]
                #    if value_of_intercept == 0:
                #        model.intercept_ = np.zeros((2,))
                #    else:
                #        model.intercept_ = np.ones((2,))
                #    model.coef_ = np.zeros((2,4008))

                #return utils.get_model_parameters(model), len(X_train), {}
                #else:


                model.fit(X_train, y_train, sample_weights)
                # add noise to gradients
                '''
                    model.intercept_
                '''
            print(f"Training finished for round {config['rnd']}")
            return utils.get_model_parameters(model), len(X_train), {}


        def evaluate(self, parameters, config):  # type: ignore
            utils.set_model_params(model, parameters)
            loss = log_loss(y_test, model.predict_proba(X_test))
            #accuracy = model.score(X_test, y_test)
            roc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
            return loss, len(X_test), { 'roc': roc}

    # Start Flower client
    fl.client.start_numpy_client("0.0.0.0:8080", client=MnistClient())
