library(pROC)
Y_true <- read.csv("C:\\Users\\§E®a·ç\\3.14\\Energy\\TRIauc\\test_Y.csv")  #dataframe
Y_true <- Y_true$class

Y_pred <- read.csv("C:\\Users\\§E®a·ç\\3.14\\Energy\\TRIauc\\NN5_prob.csv")  #dataframe
Y_pred <- Y_pred$prob
#class(Y_pred)

auc(response = Y_true, predictor =  Y_pred)
ci.auc(response = Y_true, predictor =  Y_pred)
