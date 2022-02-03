import argparse
import os

import tensorflow as tf

import flwr as fl

import common
from sklearn.metrics import log_loss,roc_auc_score,confusion_matrix

import numpy as np
import pickle as pk

# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def get_eval_fn(model):
    """Return an evaluation function for server-side evaluation."""
    # Load test data here to avoid the overhead of doing it in `evaluate` itself
    X_test, y_test = load_test_data()

    # The `evaluate` function will be called after every round
    def evaluate(parameters: fl.common.Weights):
        # Update model with the latest parameters
        model.set_weights(parameters)
        loss = log_loss(y_test, model.predict(X_test))
        #accuracy = model.score(X_test, y_test)
        y_prob = model.predict(X_test)
        roc = roc_auc_score(y_test,y_prob )
        #tn, fp, fn, tp = confusion_matrix(y_test.ravel(), np.argmax(y_prob, axis = 1).ravel(), labels=[0, 1]).ravel()
        #tpr = tp/(tp+fn)
        #pk.dump(model, open("lr_last_round_fed_cb_dp6.pkl", 'wb'))
        #joblib.dump(model,"lr_last_round_fed_cb_dp6.joblib")
        return loss, {"roc": roc}

    return evaluate

def load_test_data():
    X = np.load("/Users/sikha/0FAIRFL/Data/113p/X_test.npy", allow_pickle=True)
    Y = np.load("/Users/sikha/0FAIRFL/Data/113p/y_test.npy", allow_pickle=True)
    #X = tf.convert_to_tensor(np.asarray(X, np.float32), dtype=tf.float32)
    print(X.shape,Y.shape)
    return X, Y

def main(args) -> None:
    model = common.create_mlp_model()
    #loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    model.compile("sgd", loss="binary_crossentropy", metrics=["accuracy"])
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=args.fraction_fit,
        min_available_clients=args.num_clients,
        eval_fn=get_eval_fn(model),
        initial_parameters=fl.common.weights_to_parameters(model.get_weights()),
    )
    fl.server.start_server(strategy=strategy, config={"num_rounds": args.num_rounds})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Server Script")
    parser.add_argument("--num-clients", default=113, type=int)
    parser.add_argument("--num-rounds", default=2, type=int)
    parser.add_argument("--fraction-fit", default=1.0, type=float)
    args = parser.parse_args()
    main(args)
