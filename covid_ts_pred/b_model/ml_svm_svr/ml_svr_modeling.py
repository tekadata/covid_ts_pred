import os
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
import covid_ts_pred.c_eng.engineering as get_csv_out_path

def create_country_X_y(country, is_index=True, n_days=15, n_pred=7) -> tuple:
    """
    create_country_X_y(country) -> tuple:
    function that take in parameter:
     - a country name: `country` (string),
     - is an index (or an indicator): `is_index` (boolean) True by default,
     - a number of days to shift: `n_days` (int) 15 by default,
    and returns a tuple with:
     - a X df: `X` (DataFrame),
     - a y df: `y` (DataFrame).
    """
    if is_index == True:
        country_df = pd.read_csv(get_csv_out_path(f"index_{country}.csv"), index_col=False)
    else:
        country_df = pd.read_csv(get_csv_out_path(f"indicator_{country}.csv"))

    df = country_df['total_deaths'].copy()
    df = pd.DataFrame(df)
    #data_confirmed_deaths_days = pd.concat([data_confirmed_deaths_days,data_confirmed_deaths_days.shift(periods=1)], axis=1)
    #data_confirmed_deaths_days = pd.concat([data_confirmed_deaths_days,data_confirmed_deaths_days.shift(periods=2)], axis=1)
    #data_confirmed_deaths_days = pd.concat([data_confirmed_deaths_days,data_confirmed_deaths_days.shift(periods=4)], axis=1)

    col_names = ['total_deaths']
    # Next `n_days` timeseries shifting
    for n in range(1, n_days + 1):
        col_names.append(f'day-{day}')
        df[col_names[n]]=df['total_deaths'].shift(periods=n)

    df.columns = col_names

    df.drop(columns= 'total_deaths', inplace = True)

    y = country_df['total_deaths']
    X = country_df.copy()
    X = pd.concat([X, df], axis=1)
    X = X.drop(columns = ['Unnamed: 0','date','new_cases', 'new_deaths', 'total_deaths'])
    # y without nb days of pred
    y = y[:n_pred]
    X = pd.DataFrame(X).dropna().reset_index(drop=True)

    return  X, y

def train_test_set_ml(country, days, is_index=True) -> tuple:
    """
    train_test_set_ml_indicator(country, days) -> tuple:
    function that take in parameter:
     - a country name: `country` (string),
     - is an index (or an indicator): `is_index` (boolean) True by default,
     - a number of days: `days` (int),
    and returns a tuple with:
     - a X train scaled df: `X_train_scaled` (DataFrame),
     - a y train df: `y_train` (DataFrame),
     - a X test scaled df: `X_test_scaled` (DataFrame),
     - a y test df: `y_test` (DataFrame).
    """
    X, y = create_country_X_y(country, is_index=is_index)

    train = int(((len(X)-days)))

    X_train = X[:train]
    y_train = y[:train]

    X_test = X[train:]
    y_test = y[train:]

    scaler = MinMaxScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, y_train, X_test_scaled, y_test


def model_ml_indicator(country, days) -> tuple:
    """
    model_ml_indicator(country, days) -> tuple:
    function that take in parameter:
     - a country name: `country` (string),
     - a number of days: `days` (int),
    and returns a tuple with:
     - a grid search best estimator: `best_model` (float),
     - a grid search best params: `best_params` (float?),
     - a grid search best score: `best_score` (float).
    """
    X_train, y_train, X_test, y_test = train_test_set_ml(country, days, is_index=False)

    model = SVR()

    param={'kernel' : ('linear', 'poly', 'rbf', 'sigmoid'),'C' : [1,5,10],'degree' : [2,3,5],
        'coef0' : [0.1,0.5,1],'gamma' : ('auto','scale')}

    grid_search = GridSearchCV(model, param_grid = param, scoring= 'neg_mean_absolute_percentage_error',
                        cv = 2, n_jobs = -1, verbose = 2, refit=True)

    grid_search.fit(X_train,y_train)
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    return best_model, best_params, best_score


def model_ml_index(country, days) -> tuple:
    """
    model_ml_index(country, days) -> tuple:
    function that take in parameter:
     - a country name: `country` (string),
     - a number of days: `days` (int),
    and returns a tuple with:
     - a model: `model` (float),
     - a mape score: `score` (float).
    """
    X_train, y_train, X_test, y_test = train_test_set_ml_index(country, days, is_index=True)

    model =SVR(C=5, coef0=10, degree=8, epsilon=0.05, gamma='auto', kernel='poly')
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    score=mean_absolute_percentage_error(y_test, y_pred)

    #param={'kernel' : ('linear', 'poly', 'rbf', 'sigmoid'),'C' : [1,5,10],'degree' : [3,8],'coef0' : [0.01,10,0.5],'gamma' : ('auto','scale')},

    #grid_search = GridSearchCV(model, param_grid = param,
    #                  cv = 2, n_jobs = -1, verbose = 2)

    #grid_search.fit(X_train,y_train)
    #best_model = grid_search.best_estimator_
    #best_params = grid_search.best_params_
    #best_score = grid_search.best_score_

    return model, score


def concat(country):

    csv_path = uz.get_raw_data_path("raw_data_index", f"data_{country}")

    country_indicator = pd.read_csv(csv_path)

    X = country_indicator.drop(columns = ['date','new_cases', 'new_deaths', 'total_deaths'])
    y = country_indicator['total_deaths']

    data_confirmed_cases_days = X['total_cases']
    data_confirmed_cases_days = pd.concat([data_confirmed_cases_days,data_confirmed_cases_days.shift(periods=1)], axis=1)
    data_confirmed_cases_days = pd.concat([data_confirmed_cases_days,data_confirmed_cases_days.shift(periods=2)], axis=1)
    data_confirmed_cases_days = pd.concat([data_confirmed_cases_days,data_confirmed_cases_days.shift(periods=4)], axis=1)

    columns_names = ['total_cases']

    for day in range(1,data_confirmed_cases_days.shape[1]):
        columns_names.append(f'day-{day}')

    data_confirmed_cases_days.columns = columns_names

    X = X.merge(data_confirmed_cases_days, on='total_cases')

    scaler = MinMaxScaler()

    X_scaled = scaler.fit_transform(X)

    X_scaled = pd.DataFrame(X_scaled).dropna().reset_index(drop=True)

    return X_scaled, y

def train_test_set(country, split_train=0.8, split_val=0):

    X, y = concat(country)

    train = int((len(X)*split_train))
    val = int(len(X)*split_val)

    X_train = X[:train]
    y_train = y[:train]

    if split_val <= split_train:
        X_test = X[train:]
        y_test = y[train:]
        return X_train, y_train, X_test, y_test

    X_val = X[train:val]
    y_val = y[train:val]

    X_test = X[val:]
    y_test = y[val:]

    return X_train, y_train, X_val, y_val, X_test, y_test

def model_ml(country):

    X_train, y_train, X_test, y_test = train_test_set(country)

    model = SVR(C=1, coef0=10, degree=8, epsilon=0.05, gamma='auto', kernel='poly')

    model = model.fit(X_train, y_train)

    return model


if __name__ == "__main__":
    # 1. Load Data
    # df = load_data(DATASET_PATH)

    # 2. Data prepartion
    # df_final = prepare_data(df)

    # 3. Prediction
    # predict(df_final)
