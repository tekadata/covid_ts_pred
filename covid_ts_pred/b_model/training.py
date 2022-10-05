import pickle
import pandas as pd
import numpy as np
from covid_ts_pred.c_eng.engineering import get_best_models_path


def predict(country,  X_test, y_train, df,  y, n_days=10)-> tuple:
    """
    predict(country,  X_test, y_test, X_train, y_train, df, y, n_days=10)-> tuple:
    function that take in parameter:
     - a X test df: `X_test` (DataFrame),
     - a y train df: `y_train` (DataFrame),
     - a df: `df` (DataFrame),
     - a y df: `y` (DataFrame)
     - a number of days to shift: `n_days` (int) 10 by default,
    and returns a tuple with:
    list_pred, X_predict
     - a prediction list: `list_pred` (list),
     - a X prediction df: `X_predict` (DataFrame).
    """
    model=pickle.load(open(get_best_models_path(f'model_{country}.pkl'),'rb'))
    X_test_columns=df.drop(columns=['total_deaths','new_cases','new_deaths'])
    X_test_df = pd.DataFrame(X_test, columns=X_test_columns.columns)

    X_predict = X_test_df.reset_index(drop=True)
    for i in range(1,n_days):
        X_predict.loc[i,'containment_and_health':'total_boosters']=X_predict.loc[0,'containment_and_health':'total_boosters']
    min_num = min(y)
    max_num = max(y)
    list_pred = []
    y_val = y_train.tail(1).values[0]


    y_pred_1 = np.round(model.predict(pd.DataFrame(X_predict.loc[0]).T))

    if y_pred_1 < y_val:
        y_pred_1 = y_val
        y_pred_2 = y_val

    else:
        y_pred_2 = y_pred_1

    y_pred_scale=((y_pred_)-min_num) / ((max_num)-(min_num))
    list_pred.append(y_pred_)

    for i in range(1, (n_days - 1)):
        for j in range(i, (n_days + 1)):
            print("X_predict.loc[j, f'day-{j}']=y_pred_scale[0]", j, f'day-{j}', y_pred_scale)
            X_predict.loc[j,f'day-{j - (i - 1)}'] = y_pred_scale[0]
        y_pred_1 = np.round(model.predict(pd.DataFrame(X_predict.loc[i]).T))
        if y_pred_1[0] < y_pred_2[0]:
            y_pred_1 = y_pred_2[0]
        else:
            y_pred_2 = y_pred_1[0]
        y_pred_scale = (y_pred_1 - min_num) / (max_num - min_num)
        list_pred.append(y_pred_1)

    return list_pred, X_predict
