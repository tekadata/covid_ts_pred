import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# def scaling(X):
#     scaler = MinMaxScaler()
#     scaler.fit(X)
#     X=scaler.transform(X)
#     return X
# def train_test_split(X,y):
#     n = len(X)
#     X_train = X[0:int(n*0.8)]
#     X_test=X[int(n*0.8):]
#     y_train=y[0:int(n*0.8)]
#     y_test=y[int(n*0.8):]
#     return X_train,y_train,X_test,y_test

# def grid_search_cv(X_train,y_train,model):
#     param={'kernel' : ('linear', 'poly', 'rbf', 'sigmoid'),'C' : [1,5,10],'degree' : [3,8],'coef0' : [0.01,10,0.5],'gamma' : ('auto','scale')},

#     grid_search = GridSearchCV(model, param_grid = param, cv = 2, n_jobs = -1, verbose = 2)
#     grid_search.fit(X_train,y_train)
#     best=grid_search.best_estimator_
#     return best
