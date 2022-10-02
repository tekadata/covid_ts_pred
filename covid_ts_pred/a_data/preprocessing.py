import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from covid_ts_pred.c_eng.engineering import get_csv_out_path


def preprocessing(country, n_days=10) -> tuple:
   """
    preprocessing(country) -> tuple:
    function that take in parameter:
     - a country name: `country` (string),
     - a number of days to shift: `n_days` (int) 10 by default,
    and returns a tuple with:
     - a X test df: `X_test` (DataFrame),
     - a y test df: `y_test` (DataFrame),
     - a X train df: `X_train` (DataFrame),
     - a y train df: `y_train` (DataFrame),
     - a df: `df` (DataFrame),
     - a y df: `y` (DataFrame).
    """
    #countries=[]
    #path='data/out_csv'
    #for country in country_list:
    df=pd.read_csv(get_csv_out_path(f"index_{country}.csv"))
    df=df.set_index('date')

    # Next `n_days` timeseries shifting
    for n in range(1, n_days + 1):
        df[f'day-{n}']=df['total_deaths'].shift(periods=n)

    # Data `n_days` first time series truncking
    data_index = data_index.iloc[n_days: , :]
    print('data_index.shape', data_index.shape)
    # NA data filling
    df=df.fillna(0)
    # X and y data splitting
    X=df.drop(columns=['total_deaths','new_deaths','new_cases'])
    y=df['total_deaths']
    # X data scaling
    scaler = MinMaxScaler()
    scaler.fit(X)
    X=scaler.transform(X)
    # X length
    l_X = len(X)
    # test length = 50
    l_X_test = 15
    # train length
    l_X_train = l_X - l_X_test
    # Train & test set data splitting
    X_train = X[0:int(l_X_train)]
    X_test=X[int(l_X_train):]
    y_train=y[0:int(l_X_train)]
    y_test=y[int(l_X_train):]
    print('X_train.shape', X_train.shape, 'X_test.shape', X_test.shape, 'y_train.shape', y_train.shape,' y_test.shape',  y_test.shape)

    return X_test, y_test, X_train, y_train, df, y


def scale_country_index(country) -> tuple:
    """
    scale_country_index(country) -> tuple:
    function that take in parameter:
     - a country name: `country` (string),
    and returns a tuple with:
     - a X scaled df: `X_scaled` (DataFrame),
     - a y df: `y` (DataFrame).
    """

    csv_path = os.path.join(get_csv_out_path(), f"index_{country}.csv")

    country_indicator = pd.read_csv(csv_path)

    X = country_indicator.drop(columns = ['date', 'new_cases', 'new_deaths', 'total_deaths'])

    y = country_indicator['total_deaths']

    scaler = MinMaxScaler()

    X_scaled = scaler.fit_transform(X)

    return X_scaled, y


def train_test_set(country, split_train=0.8, split_val=0) -> tuple:
    """
    train_test_set(country, split_train=0.8, split_val=0, switch_to_index=False) -> tuple:
    function that take in parameter:
     - a country name: `country` (string),
     - a split train df, set to 80% by default: `split_train` (DataFrame),
     - a split validation df, set to 0% by default: `split_val`(DataFrame),
     - a switch to index boolean, set to False by default: `switch_to_index` (boolean),
    and returns a tuple with:
     - a X train df: `X_train` (DataFrame),
     - a y train df: `y_train` (DataFrame),
     - a X validation df: `X_val` (DataFrame),
     - a y validation df: `y_val` (DataFrame),
     - a X test df: `X_test` (DataFrame),
     - a y test df: `y_test` (DataFrame).
    """
    X, y = scale_country_index(country)


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
