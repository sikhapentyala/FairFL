import flwr as fl
#import utils_ads as utils
import common
from sklearn.metrics import log_loss,roc_auc_score,confusion_matrix
from sklearn.linear_model import LogisticRegression
from typing import Dict
import numpy as np
import pickle as pk
import joblib
import os
import tensorflow as tf
#import lightgbm as lgb
#RANDOM_SEED = 47568
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
RANDOM_SEED = 1000
#seed(47568)
#tf.random.set_random_seed(seed_value)
os.environ['PYTHONHASHSEED']=str(RANDOM_SEED)
#random.seed(47568)
tf.random.set_seed(RANDOM_SEED)

#SAVE = False

np.random.seed(RANDOM_SEED)




def fit_round(rnd: int) -> Dict:
    config = {"rnd": rnd}
    return config

def evaluate_config(rnd: int):
    """Return evaluation configuration dict for each round.

    Perform five local evaluation steps on each client (i.e., use five
    batches) during rounds one to three, then increase to ten local
    evaluation steps.
    """
    return {"rnd": rnd}


def get_eval_fn(model: LogisticRegression):
    """Return an evaluation function for server-side evaluation."""

    # Load test data here to avoid the overhead of doing it in `evaluate` itself
    X_test, y_test = load_test_data()

    # The `evaluate` function will be called after every round
    def evaluate(parameters: fl.common.Weights):
        # Update model with the latest parameters
        model.set_weights(parameters)
        loss = log_loss(y_test, model.predict(X_test))
        y_prob = model.predict(X_test)
        roc = roc_auc_score(y_test,y_prob[:,1] )
        joblib.dump(model,"lr_last_round_fl.joblib")
        return loss, {"roc": roc}

    return evaluate

def load_test_data():
    X = np.load("/home/sikha/data/X_test.npy", allow_pickle=True)
    Y = np.load("/home/sikha/data/y_test.npy", allow_pickle=True)
    return X, Y


# Start Flower server for five rounds of federated learning
if __name__ == "__main__":

    model = common.create_mlp_model()
    #loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    model.compile("sgd", loss="binary_crossentropy", metrics=[tf.keras.metrics.Recall()])


    strategy = fl.server.strategy.FedAvg(
        min_available_clients=109,
        eval_fn=get_eval_fn(model),
        on_fit_config_fn=fit_round,
        fraction_fit=1.0, #1
        fraction_eval=1.0, #1
        min_fit_clients=109,
        min_eval_clients=109, #6
        #on_evaluate_config_fn=evaluate_config,
        initial_parameters=fl.common.weights_to_parameters(model.get_weights()),
    )
    fl.server.start_server( strategy=strategy, config={"num_rounds": 800})

