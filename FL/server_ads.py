import flwr as fl
import utils_ads as utils
from sklearn.metrics import log_loss,roc_auc_score,confusion_matrix
from sklearn.linear_model import LogisticRegression
from typing import Dict
import numpy as np
import pickle as pk
import joblib
import lightgbm as lgb
RANDOM_SEED = 47568
#0.5848141880565764 3.4476200353793947 dp1
#'0':0.583175640805092 ,'1':3.5056876938986554}
#0.5829415905233408 3.5141693500517905 dp2
#0.5832284802792301 3.503779465409607 dp3
#0.5831791167067011 3.505562091763107 dp4
#0.5821571993759853 3.542946958986477 dp5
# 0.5836890562169341 3.4872484085847937 dp6
def fit_round(rnd: int) -> Dict:
    """Send round number to client."""
    # Send global count to clients
    config = {"rnd": rnd, '0':0.5836890562169341 ,'1':3.4872484085847937}
    # how to recieve and aggregate weights
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
        utils.set_model_params(model, parameters)
        loss = log_loss(y_test, model.predict_proba(X_test))
        #accuracy = model.score(X_test, y_test)
        y_prob = model.predict_proba(X_test)
        roc = roc_auc_score(y_test,y_prob[:,1] )
        #tn, fp, fn, tp = confusion_matrix(y_test.ravel(), np.argmax(y_prob, axis = 1).ravel(), labels=[0, 1]).ravel()
        #tpr = tp/(tp+fn)
        pk.dump(model, open("lr_last_round_fed_cb_dp6.pkl", 'wb'))
        joblib.dump(model,"lr_last_round_fed_cb_dp6.joblib")
        return loss, {"roc": roc}

    return evaluate

def load_test_data():
    X = np.load("/Users/sikha/0FAIRFL/Data/113p/X_test.npy", allow_pickle=True)
    Y = np.load("/Users/sikha/0FAIRFL/Data/113p/y_test.npy", allow_pickle=True)
    #X = tf.convert_to_tensor(np.asarray(X, np.float32), dtype=tf.float32)
    print(X.shape,Y.shape)
    return X, Y


# Start Flower server for five rounds of federated learning
if __name__ == "__main__":

    # Receive DP-counts from various clients




    model = LogisticRegression(
        random_state = RANDOM_SEED, solver='sag',
        penalty="l2",#class_weight={0: 0.58, 1: 3.50},
        #max_iter=1,  # local epoch
        warm_start=True,  # prevent refreshing weights when fitting
    )
    
    #model = lgb.LGBMClassifier(learning_rate=0.1,class_weight='balanced',max_depth=5,  random_state = RANDOM_SEED)

    utils.set_initial_params(model)


    strategy = fl.server.strategy.FedAvg(
        min_available_clients=113,
        eval_fn=get_eval_fn(model),
        on_fit_config_fn=fit_round,
        fraction_fit=1.0, #1
        fraction_eval=1.0, #1
        min_fit_clients=113,
        min_eval_clients=113, #6
        #on_evaluate_config_fn=evaluate_config,
        #initial_parameters=fl.common.weights_to_parameters(model.get_weights()),
    )
    fl.server.start_server("0.0.0.0:8080", strategy=strategy, config={"num_rounds": 400})
