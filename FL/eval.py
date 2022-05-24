
import sys

import numpy as np
import pandas as pd


import os


import random
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score

from numpy.random import seed


RANDOM_SEED = 42
seed(RANDOM_SEED)
#tf.random.set_random_seed(seed_value)
os.environ['PYTHONHASHSEED']=str(RANDOM_SEED)
random.seed(RANDOM_SEED)

pd.set_option('display.max_rows', None)
SAVE = False

np.random.seed(RANDOM_SEED)

print(f"Using random seed, {RANDOM_SEED}")
DATA_PATH = '/home/sikha/data/'

import pickle5 as pickle
with open(DATA_PATH+"testdf.pkl", "rb") as fh:
    test_df = pickle.load(fh)
#import pickle
#test_df = pd.read_pickle(DATA_PATH+"testdf.pkl")

SENS_ATTR = 'Female'
# gender column OHE

LABEL_NAME = 'Rating'
PROT_ATTR_NAME = 'Female'
UNPROT_ATTR_NAME = 'Male'

def load_test_data():
    X = np.load(DATA_PATH+"X_test.npy", allow_pickle=True)
    Y = np.load(DATA_PATH+"y_test.npy", allow_pickle=True)
    return X, Y

import joblib
# Normal
X_test, y_test = load_test_data()



flmodel = joblib.load("lr_last_round_fl.joblib")
fl_y_prob = flmodel.predict(X_test)
print("ROC AUC: ", roc_auc_score(y_test.ravel(), fl_y_prob[:,1]))
print("Accuracy: ", accuracy_score(y_test,np.argmax(fl_y_prob, axis = 1).ravel()))


test_df['pred_fl_bal'] = np.argmax(fl_y_prob, axis = 1)
test_df['pred_fl_bal'] = test_df['pred_fl_bal'].astype('int')

prot_df = test_df[test_df[PROT_ATTR_NAME] == 1][[LABEL_NAME,'pred_fl_bal']]
tn, fp, fn, tp = confusion_matrix(prot_df[[LABEL_NAME]].values.ravel(),prot_df[['pred_fl_bal']].values.ravel(), labels=[0, 1]).ravel()
prot_tpr = tp/(tp+fn)
prot_fpr = fp / (fp + tn)
prot_dp = (tp + fp) / prot_df.shape[0]
print(PROT_ATTR_NAME,":", tp/(tp+fn))

unprot_df = test_df[test_df[UNPROT_ATTR_NAME] == 1][[LABEL_NAME,'pred_fl_bal']]
tn, fp, fn, tp = confusion_matrix(unprot_df[[LABEL_NAME]].values.ravel(),unprot_df[['pred_fl_bal']].values.ravel(), labels=[0, 1]).ravel()
unprot_tpr = tp/(tp+fn)
unprot_fpr = fp / (fp + tn)
unprot_dp = (tp + fp) / unprot_df.shape[0]
print("DI:", max(prot_tpr / unprot_tpr, unprot_tpr / prot_tpr))
print("EOP:", abs(prot_tpr - unprot_tpr))
print("Avg EP diff:", 0.5 * (abs(prot_tpr - unprot_tpr) + abs(prot_fpr - unprot_fpr)))
print("SPD:", abs(prot_dp - unprot_dp))
#print(UNPROT_ATTR_NAME,":", tp/(tp+fn))

#print("DI:", max(prot_tpr/unprot_tpr, unprot_tpr/prot_tpr))
