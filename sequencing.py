import numpy as np
import pandas as pd
from covid_ts_pred.b_viz.b_preproc.preproc import train_test_set

def subsample_sequence_2(X, y, X_len, y_len) -> pd.DataFrame:
    """
    Given the initial arrays `X` and `y`, return shorter array sequences.
    This shorter sequence should be selected at random
    """
    X_y_len = X_len + y_len
    # print('X_len', X_len,  'y_len',   y_len)
    # print('X.shape[0]', X.shape[0], ' >= X_y_len ', X_y_len)
    if X.shape[0] >= X_y_len:
        last_possible = X.shape[0] - X_y_len
    else:
        last_possible = X.shape[0]
        # print('X_y_len = ?', X.shape[0])
    random_start = np.random.randint(0, last_possible)
    # X start and y end
    X_sample = X[random_start : random_start + X_len]
    y_sample = y[random_start + X_len : (random_start + X_y_len)]
    # print("X[random_start : random_start + X_len]   -> ", f"X[{random_start} : {random_start + X_len}]")
    # print("y[random_start : random_start + X_y_len] -> ", f"y[{random_start} : {(random_start + X_y_len)}]")

    return np.array(X_sample), np.array(y_sample)

def subsample_sequence(df, length):
    """
    Given the initial dataframe `df`, return a shorter dataframe sequence of length `length`.
    This shorter sequence should be selected at random.
    """

    last_possible = df.shape[0] - length

    random_start = np.random.randint(0, last_possible)
    df_sample = df[random_start: random_start+length]

    return df_sample


def get_X_y(df, n_sequences, length, feature='VNM') -> tuple:
    '''Return a list of samples (`X`,`y`)'''
    X, y = [], []

    for i in range(n_sequences):
        (xi, yi) = split_subsample_sequence(df, length, feature=feature)
        X.append(xi)
        y.append(yi)

    X = np.array(X)
    y = np.array(y)

    return X, y


def get_X_y_2(X, y, X_len, y_len, n_sequences) -> tuple:
    '''Return a list of samples (X, y)'''
    X_list, y_list = [], []

    for i in range(n_sequences):
        (xi, yi) = subsample_sequence_2(X, y, X_len=X_len, y_len=y_len)
        X_list.append(xi)
        y_list.append(yi)

    X = np.array(X_list)
    y = np.array(y_list)

    return X, y




def compute_means(X, df_mean):
    '''utils'''
    # Compute means of X
    means = X.mean()

    # Case if ALL values of at least one feature of X are NaN, then reaplace with the whole df_mean
    if means.isna().sum() != 0:
        means.fillna(df_mean, inplace=True)

    return means

def split_subsample_sequence(df, length, df_mean=None):
    """Return one single sample (Xi, yi) containing one sequence each of length `length`"""
    features_names = ['TEMP', 'DEWP', 'PRES', 'Ir', 'Is', 'Iws']

    # Trick to save time during the recursive calls
    if df_mean is None:
        df_mean = df[features_names].mean()

    df_subsample = subsample_sequence(df, length).copy()

    # Let's drop any row without a target! We need targets to fit our model
    df_subsample.dropna(how='any', subset=['pm2.5'], inplace=True)

    # Create y_sample
    if df_subsample.shape[0] == 0: # Case if there is no targets at all remaining
        return split_subsample_sequence(df, length, df_mean) # Redraw by recursive call until it's not the case anymore
    y_sample = df_subsample[['pm2.5']]

    # Create X_sample
    X_sample = df_subsample[features_names]
    if X_sample.isna().sum().sum() !=0:  # Case X_sample has some NaNs
        X_sample = X_sample.fillna(compute_means(X_sample, df_mean))

    return np.array(X_sample), np.array(y_sample)
