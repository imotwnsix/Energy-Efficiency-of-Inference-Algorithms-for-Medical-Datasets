# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 22:13:42 2020

@author: 余家瑞
"""
from joblib import load
import pandas as pd
from sklearn import metrics
import numpy as np
import csv


X_test = pd.read_csv("X_test.txt")
Y_test = pd.read_csv("Y_test.txt")


algo = input('請輸入演算法：')
model = load(f'{algo}_MRSA_paper.joblib')
pred = model.predict(X_test)
prob = np.array(model.predict_proba(X_test))

# 檢查用：記得probability和model class要對齊。
# print(model.classes_)
# print(prob)


if algo == 'XGB':
    Rprob = prob[:,1]
    Y_test.replace(to_replace="S", value=0, inplace=True)
    Y_test.replace(to_replace="R", value=1, inplace=True) 
    print(f"accuracy score: {metrics.accuracy_score(pred,Y_test)}") #accuracy 似乎可以使用dataframe計算。
else:
    Rprob = prob[:,0]
    print(f"accuracy score: {metrics.accuracy_score(pred,Y_test)}")
    Y_test.replace(to_replace="S", value=0, inplace=True)
    Y_test.replace(to_replace="R", value=1, inplace=True)


# print(f"proba:{Rprob}")
ans = Y_test['Label'].values #計算AUC之前先把dataframe換成np array
# print(f"Y_test:{ans}")
# print(Rprob, file = open(f"{algo}_Rprob.csv","a", encoding="utf-8"))
# print(ans, file = open(f"{algo}_ans.csv","a", encoding="utf-8"))

# with open(f"{algo}_ans.csv", "w", newline = "", errors = "ignore") as file:
#     writer = csv.writer(file)
#     writer.writerow(list(ans))

with open(f"{algo}_ans.csv", "a", newline = "", errors = "ignore") as file:
    writer = csv.writer(file)
    writer.writerow(['answer'])
    for num in list(ans):
        writer.writerow([num])
    
with open(f"{algo}_Rprob.csv", "a", newline = "", errors = "ignore") as file:
    writer = csv.writer(file)
    writer.writerow(['prob'])
    for num in list(Rprob):
        writer.writerow([num])
    
# np.savetxt("test.csv", a,fmt="%.0f", delimiter=",")
# np.savetxt("test.csv", a,fmt="%.0f", delimiter=",")
    
print(f'AUC score:{metrics.roc_auc_score(ans, Rprob)}')

# print(list(ans))
# print(list(Rprob))
# print(len(list(ans)))
# print(len(list(Rprob)))
    









# # THE MODEL for XGB以外
# model = joblib.load('kNN_MRSA_Tune.pkl')
# pred = model.predict(X_test) #for accuracy
# pred_p = model.predict_proba(X_test) # for AUC
# pred_p = pd.DataFrame(pred_p)
# Y_ROC = pred_p[0]



# # THE MODEL for XGB
# model = load('XGB_MRSA_Tune.joblib.dat')
# Y_ROC = model.predict(X_test)
# pred = Y_ROC
# Y_test.replace(to_replace="S", value=0, inplace=True)
# Y_test.replace(to_replace="R", value=1, inplace=True)




# # counting accuracy
# print(f"accuracyscore: {metrics.accuracy_score(pred,Y_test)}")
# print(metrics.classification_report(pred,Y_test))



# # 以下為計算AUC
# # Y_test dataframe換成0 1
# Y_test.replace(to_replace="S", value=0, inplace=True)
# Y_test.replace(to_replace="R", value=1, inplace=True)
# fpr, tpr, thresholds = metrics.roc_curve( Y_test,Y_ROC)
# print(f"AUC: {metrics.auc(fpr, tpr)}")

