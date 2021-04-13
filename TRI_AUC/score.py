# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 00:01:27 2020

@author: 余家瑞
"""
import numpy as np
from joblib import load
import pandas as pd
from sklearn.metrics import roc_curve, auc, accuracy_score
import csv
X_test = pd.read_csv("test_X.csv")
Y_test = pd.read_csv("test_Y.csv")

algo = input('請輸入演算法：')
model = load(f'{algo}_Tri_smote.joblib')
pred = model.predict(X_test)

if algo != 'SVM':
    prob = np.array(model.predict_proba(X_test))
    score = prob[:,1] #probability of class 1
else:
    score = model.decision_function(X_test)
    
fpr, tpr, _ = roc_curve(Y_test, score)
AUC = auc(fpr, tpr)

print(f"accuracy:{accuracy_score(pred,Y_test)}")
print(f"AUC:{AUC}")

with open(f"{algo}_prob.csv", "a", newline = "", errors = "ignore") as file:
    writer = csv.writer(file)
    writer.writerow(['prob'])
    for num in list(score):
        writer.writerow([num])