# Import dependencies
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU, Flatten
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import RMSprop

from covid_ts_pred.b_viz.b_preproc.preproc import train_test_set
from covid_ts_pred.b_model.db_ml_models.sequencing import get_X_y_2

## TENSORFLOW & RNN MODEL

### Model Training
def train_rnn_model(model, X_train, y_train, X_val=np.array(1), y_val=np.array(1), validation_split=0, patience=20, epochs=200, batch_size=32):
    """ function that train a RNN model with hyperparameters:
    - patience by default 2 to early stop
    - epochs by default 200 to train over several epochs
    - valisation data by default (X_val, y_val)=(0, 0) in case of auto split
    """
    es = EarlyStopping(monitor = 'val_loss',
                    patience = patience,
                    verbose = 0,
                    restore_best_weights = True)
    # The fit
    if validation_split > 0:
        history =  model.fit(X_train,
                y_train,
                # Auto split for validation data
                # [print(f'validation_data=(X_val, y_val),') if (X_val!=0 or y_val!=0) else print(f'validation_split=0.1,')],
                validation_split=validation_split, # Auto split for validation data
                batch_size = batch_size,
                epochs = epochs,
                callbacks = [es],
                verbose = 0)
    else:
        history =  model.fit(X_train,
                y_train,
                # Auto split for validation data
                # [print(f'validation_data=(X_val, y_val),') if (X_val!=0 or y_val!=0) else print(f'validation_split=0.1,')],
                validation_data=(X_val, y_val),
                batch_size = batch_size,
                epochs = epochs,
                callbacks = [es],
                verbose = 0)
    return history

### Model Architecturing
def arch_rnn_model_1(X_train, n_pred):
    """ function that create a RNN model architeture with 2 LSTM layers :
    - BEST MAPE: 6.837233066558838 with model #1, nb obs: 65, lr: 0.001, n_pred = 3
    - BEST MAPE: 9.387785911560059 with model #1, nb obs: 60, lr: 0.001, n_pred = 10
    """
    # print('input_shape', X_train.shape)
    rnn_model = Sequential()
    # rnn_model.add(normalizer) # Using the Normalization layer to standardize the datapoints during the forward pass
    # Input len(train) (input_shape=(?,?))
    rnn_model.add(LSTM(units=20, activation='tanh', input_shape=(X_train.shape[-2],X_train.shape[-1]), return_sequences=True))  ## , input_shape=(?,?))) without a Normalizer layer
    # 2nd layer
    rnn_model.add(LSTM(units=10, activation='tanh'))  ## , input_shape=(?,?))) without a Normalizer layer
    # rnn_model.add(layers.Dropout(0.3)) ## if RNN model over-fit
    rnn_model.add(Dense(10, activation = 'relu')) ## add 1 or more 'relu' layers
    rnn_model.add(Dense(n_pred, activation = 'linear'))

    return rnn_model

def arch_rnn_model_2(X_train, n_pred):
    """ function that create a RNN model architeture with 3 LSTM layers:
    - LSTM
    - Dense
    - 3rd model layers architecture (simple -> complex) (less data -> more data) (print(loss) function check lecture)
    > LSTM
    """
    # print('input_shape', X_train.shape)
    rnn_model = Sequential()
    # rnn_model.add(normalizer) # Using the Normalization layer to standardize the datapoints during the forward pass
    # Input len(train) (input_shape=(?,?))
    rnn_model.add(LSTM(units=30, activation='tanh', input_shape=(X_train.shape[-2],X_train.shape[-1]), return_sequences=True))  ## , input_shape=(?,?))) without a Normalizer layer
    # 2nd layer
    rnn_model.add(LSTM(units=20, activation='tanh', return_sequences=True))  ## , input_shape=(?,?))) without a Normalizer layer
    # 3rd layer
    rnn_model.add(LSTM(units=10, activation='tanh'))  ## , input_shape=(?,?))) without a Normalizer layer
    # rnn_model.add(layers.Dropout(0.3)) ## if RNN model over-fit
    rnn_model.add(Dense(10, activation = 'relu')) ## add 1 or more 'relu' layers
    rnn_model.add(Dense(n_pred, activation = 'linear'))

    return rnn_model

def arch_rnn_model_3(X_train, n_pred):
    """ function that create a RNN model architeture with 3 LSTM layers with a last RELU layer:
    - BEST MAPE: 6.688204288482666 with model #3, nb obs: 60, lr: 0.001, n_pred=10, c='FRA'
    - BEST MAPE: 18.718429565429688 with model #3, nb obs: 60, lr: 0.001, n_pred=10, c='FRA'
    """
    # print('input_shape', X_train.shape)
    rnn_model = Sequential()
    # rnn_model.add(normalizer) # Using the Normalization layer to standardize the datapoints during the forward pass
    # Input len(train) (input_shape=(?,?))
    rnn_model.add(LSTM(units=30, activation='tanh', input_shape=(X_train.shape[-2],X_train.shape[-1]), return_sequences=True))  ## , input_shape=(?,?))) without a Normalizer layer
    # rnn_model.add(layers.Dropout(0.3)) ## if RNN model over-fit
    # 2nd layer
    rnn_model.add(LSTM(units=20, activation='tanh', return_sequences=True))  ## , input_shape=(?,?))) without a Normalizer layer
    # 3rd layer
    rnn_model.add(LSTM(units=10, activation='relu'))  ## , input_shape=(?,?))) without a Normalizer layer
    # rnn_model.add(layers.Dropout(0.3)) ## if RNN model over-fit
    rnn_model.add(Dense(10, activation = 'relu')) ## add 1 or more 'relu' layers
    rnn_model.add(Dense(n_pred, activation = 'linear'))

    return rnn_model

def arch_rnn_model_4(X_train, n_pred):
    """ function that create a RNN model architeture with layers with a last RELU layer:
    - BEST MAPE: 51.2643356323242 with model #3, nb obs: 65, lr: 0.001, n_pred=7
    - BEST MAPE: 6.688204288482666 with model #3, nb obs: 60, lr: 0.001, n_pred=10, c='FRA'
    - BEST MAPE: 15.1573486328125 with model #3, nb obs: 60, lr: 0.001, n_pred=10, c='DEU'
    - BEST MAPE: 15.767879486083984 with model #8, nb obs: 60, lr: 0.001, n_pred=10, c='USA'
    - BEST MAPE: 15.767879486083984 with model #8, nb obs: 60, lr: 0.001, n_pred=10, c='USA'
    - BEST MAPE: 16.32421875 with model #8, nb obs: 60, lr: 0.001, n_pred=10, c='UK'
    - BEST MAPE: 17.89015769958496 with model #8, nb obs: 60, lr: 0.001, n_pred=10, c='USA'
    - BEST MAPE: 25.942426681518555 with model #8, nb obs: 60, lr: 0.001, n_pred=10, c='India'
    - BEST MAPE: 9.994515419006348 with model #3, nb obs: 60, lr: 0.001, n_pred=10, c='Brazil'
    """
    # print('input_shape', X_train.shape)
    rnn_model = Sequential()
    # rnn_model.add(normalizer) # Using the Normalization layer to standardize the datapoints during the forward pass
    # Input len(train) (input_shape=(?,?))
    rnn_model.add(LSTM(units=30, activation='tanh', input_shape=(X_train.shape[-2],X_train.shape[-1]), return_sequences=True))  ## , input_shape=(?,?))) without a Normalizer layer
    # 2nd layer
    rnn_model.add(LSTM(units=20, activation='tanh', return_sequences=True))  ## , input_shape=(?,?))) without a Normalizer layer
    # 3rd layer
    rnn_model.add(LSTM(units=10, activation='tanh'))  ## , input_shape=(?,?))) without a Normalizer layer
    # Dense layer
    rnn_model.add(Dense(10, activation = 'relu')) ## add 1 or more 'relu' layers
    rnn_model.add(layers.Dropout(0.3)) ## if RNN model over-fit
    rnn_model.add(Dense(n_pred, activation = 'linear'))

    return rnn_model

# set_params() function
def model_run(country_name, n_seq=200, n_obs=[70], n_feat=20, n_pred=1, split_train=0.8, split_val=0, learning_rates=[0.001]):
    n_seq_val = n_seq // 5 # number of sequences in test set ?
    n_seq_test = n_seq // 10 # number of sequences in test set ?
    #### Split the dataset into training, validation and test datas
    X_train, y_train, X_val, y_val, X_test, y_test = train_test_set(country_name, split_train=split_train, split_val=split_val)
    best_n_obs = [] ; best_lr = [] ; best_model = [] ; best_MAPE = MAPE = []
    models = range(1, 5)
    # print('models', models)
    for n_o in n_obs:
        for model in models:
            for learning_rate in learning_rates:

                ### Train Splitting with sequenced training, validation and test datas
                #### for train data:
                X_train_seq, y_train_seq = get_X_y_2(X_train, y_train, X_len=n_o, y_len=n_pred, n_sequences=n_seq)
                #### for val data:
                X_val_seq, y_val_seq = get_X_y_2(X_val, y_val, X_len=n_o, y_len=n_pred, n_sequences=n_seq_val)
                #### for test data:
                X_test_seq, y_test_seq = get_X_y_2(X_test, y_test, X_len=n_o, y_len=n_pred, n_sequences=n_seq_test)

                # 1. The Architecture
                if model == 1:
                    rnn_model = arch_rnn_model_1(X_train=X_train_seq, n_pred=n_pred)
                elif model == 2:
                    rnn_model = arch_rnn_model_2(X_train=X_train_seq, n_pred=n_pred)
                elif model == 3:
                    rnn_model = arch_rnn_model_3(X_train=X_train_seq, n_pred=n_pred)
                if model == 4:
                    rnn_model = arch_rnn_model_4(X_train=X_train_seq, n_pred=n_pred)
                #elif model == 5:
                #    rnn_model = arch_rnn_model_5(X_train=X_train_seq, n_pred=n_pred)
                #elif model == 6:
                #    rnn_model = arch_rnn_model_6(X_train=X_train_seq, n_pred=n_pred)
                #elif model == 7:
                #   rnn_model = arch_rnn_model_7(X_train=X_train_seq, n_pred=n_pred)
                #elif model == 8:
                #    rnn_model = arch_rnn_model_8(X_train=X_train_seq, n_pred=n_pred)

                # 2. Compiling with 'rmsprop' rather than 'adam' (recommended)
                optimizer = RMSprop(
                                learning_rate=learning_rate,
                                rho=0.9,
                                momentum=0.0,
                                epsilon=1e-07,
                                centered=False
                            )
                rnn_model.compile(loss='mse',
                              optimizer= optimizer, # optimizer='rmsprop'    <- adapt learning rate
                                 metrics='mape')  # Recommended optimizer for RNNs

                # 3. Training
                history = train_rnn_model(model=rnn_model, X_train=X_train_seq, y_train=y_train_seq, X_val=X_val_seq, y_val=y_val_seq, epochs=400, patience=7)
                MAPE = list(history.history['mape'])
                # print('MAPE', MAPE)
                if len(MAPE) > 0:
                    if min(MAPE) > 0:
                        best_MAPE.append(min(MAPE))
                        best_lr.append(learning_rate)
                        best_model.append(model)
                        best_n_obs.append(n_o)
                        print('n_seq\t\t', n_seq, '\nn_seq_val\t', n_seq_val, '\nn_seq_test\t', n_seq_test, '\nn_obs\t\t', n_o, '\nn_feat\t\t', n_feat, '\nn_pred\t\t', n_pred)
                        print('X_train\t', X_train.shape, '\t->\ty_train\t', y_train.shape, '\nX_val\t', X_val.shape, '\t->\ty_val\t', y_val.shape, '\nX_test\t', X_test.shape, '\t->\ty_test\t', y_test.shape)
                        print('X_train_seq', X_train_seq.shape, '\t->\ty_train_seq\t', y_train_seq.shape, '\nX_val_seq', X_val_seq.shape, '\t->\ty_val_seq\t', y_val_seq.shape, '\nX_test_seq', X_test_seq.shape, '\t->\ty_test_seq\t', y_test_seq.shape)
                        print("rnn_model.summary()", rnn_model.summary())
                        print("MODEL #",model)
                        print("NB OF OBSERVATIONS TO TRAIN:", n_o)
                        print("LEARNING RATE:", learning_rate)
                        print("MIN MAPE:", min(MAPE))
                        plt.plot(history.history['mape'],label='MAPE', linewidth=3)
                        plt.plot(history.history['val_mape'], label='val MAPE')
                        plt.show();

                        if len(best_MAPE) > 0:
                            min_best_MAPE = min(best_MAPE)
                            if min_best_MAPE > 0:
                                print("BEST MAPE:", min_best_MAPE)
                                # print(best_lr, best_MAPE, best_model, best_n_obs)
                                [print(f"with model #{best_model[i]}, nb obs: {best_n_obs[i]}, lr: {best_lr[i]}") for i, v in enumerate(best_MAPE) if v == min_best_MAPE]


def model_run_2(data, country_name, n_seq=200, n_obs=[70], n_feat=20, n_pred=1, split_train=0.8, split_val=0, learning_rates=[0.001]):
    n_seq_val = n_seq // 5 # number of sequences in test set ?
    n_seq_test = n_seq // 10 # number of sequences in test set ?
    print('n_seq\t\t', n_seq, '\nn_seq_val\t', n_seq_val, '\nn_seq_test\t', n_seq_test, '\nn_obs\t\t', n_obs, '\nn_feat\t\t', n_feat)
    for n_o in n_obs:
        for learning_rate in learning_rates:
            print("NB OF OBSERVATIONS TO TRAIN:", n_o)
            print("LEARNING RATE:", learning_rate)
            ### Train Splitting
            X = data.drop(columns = ['date','new_cases', 'new_deaths', 'total_deaths'])
            y = X['total_deaths']
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X)
            train = int((len(X_scaled)*split_train))
            val = int(len(X_scaled)*split_val)

            X_train = X[:train]
            y_train = y[:train]

            if split_val <= split_train:
                X_test = X[train:]
                y_test = y[train:]
            else:
                X_val = X[train:val]
                y_val = y[train:val]
                X_test = X[val:]
                y_test = y[val:]

            print('X_train.shape\t', X_train.shape, '\t->\ty_train shape\t', y_train.shape, '\nX_val.shape\t', X_val.shape, '\t->\ty_val shape\t', y_val.shape, '\nX_test.shape\t', X_test.shape, '\t->\ty_test shape\t', y_test.shape)
            #### Create sequences (`X`,`y`, `X_len`, `y_len`)
            #### for train data:
            X_train, y_train = get_X_y_2(X_train, y_train, X_len=n_o, y_len=n_pred, n_sequences=n_seq)
            print('n_seq / n_obs / n_feat', n_seq, n_o, n_feat, '\nX_train.shape', X_train.shape, 'y_train.shape', y_train.shape)
            #### for val data:
            X_val, y_val = get_X_y_2(X_val, y_val, X_len=n_o, y_len=n_pred, n_sequences=n_seq_val)
            print('n_seq_val / n_obs / n_feat', n_seq_val, n_o, n_feat, '\nX_val.shape', X_val.shape, 'y_val.shape', y_val.shape)
            #### for test data:
            X_test, y_test = get_X_y_2(X_test, y_test, X_len=n_o, y_len=n_pred, n_sequences=n_seq_test)
            print('n_seq_test / n_obs / n_feat', n_seq_test, n_o, n_feat, '\nX_test.shape', X_test.shape, 'y_test.shape', y_test.shape)
            # Check data before training
            print('Check data before training')
            print('X_train.shape\t', X_train.shape, '\t->\ty_train shape\t', y_train.shape, '\nX_val.shape\t', X_val.shape, '\t->\ty_val shape\t', y_val.shape, '\nX_test.shape\t', X_test.shape, '\t->\ty_test shape\t', y_test.shape)
            print('type(X_train)\t', type(X_train), '\t->\ttype(y_train)\t', type(y_train), '\ntype(X_val)\t', type(X_val), '\t->\ttype(y_val)\t', type(y_val), '\ntype(X_test)\t', type(X_test), '\t->\ttype(y_test)\t', type(y_test))
            print('n_pred', n_pred)
            # dim_one=n_seq, dim_two = int(n_o) ; dim_three = int(n_feat) + int(n_pred) ; n_pred=int(n_pred)
            # 1. The Architecture
            rnn_model = arch_rnn_model(X_train=X_train, n_pred=n_pred)


            # 2. Compiling with 'rmsprop' rather than 'adam' (recommended)
            optimizer = RMSprop(
                            learning_rate=learning_rate,
                            rho=0.9,
                            momentum=0.0,
                            epsilon=1e-07,
                            centered=False
                        )
            rnn_model.compile(loss='mse',
                          optimizer= optimizer, # optimizer='rmsprop'    <- adapt learning rate
                             metrics='mape')  # Recommended optimizer for RNNs
            print("rnn_model.summary()", rnn_model.summary())
            print("NB OF OBSERVATIONS TO TRAIN:", n_o)
            print("LEARNING RATE:", learning_rate)
            # 3. Training
            history = train_rnn_model(model=rnn_model, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, epochs=400, patience=7)
            plt.plot(history.history['mape'])
            plt.plot(history.history['val_mape'])
            plt.show();


def generate_country_code(country):
    code=df_cases_raw[df_cases_raw['country_name']==country]['country_code']
    code=code.iloc[0]
    return code


def get_RNN_model_API(country_name='France', return_test=False) -> pd.DataFrame:
    """ function that return the y dataframe predicted for a given country
    """
    if len(country_name) > 3:

        n_seq=200 ; n_obs=60 ; n_feat=20
        n_seq_test = n_seq // 10 ; n_seq_val = n_seq // 5 ;
        n_pred=10 ; split_train=0.7 ; split_val=0.9 ; learning_rate=0.001

        #### Split the dataset into training, validation and test datas NOW! NOW!
        X_train, y_train, X_val, y_val, X_test, y_test = train_test_set(country_name, split_train=split_train, split_val=split_val)

        ### Train Splitting with sequenced training, validation and test datas
        #### for train data:
        X_train_seq, y_train_seq = get_X_y_2(X_train, y_train, X_len=n_obs, y_len=n_pred, n_sequences=n_seq)
        #### for val data:
        X_val_seq, y_val_seq = get_X_y_2(X_val, y_val, X_len=n_obs, y_len=n_pred, n_sequences=n_seq_val)
        #### for test data:
        X_test_seq, y_test_seq = get_X_y_2(X_test, y_test, X_len=n_obs, y_len=n_pred, n_sequences=n_seq_test)

        # 1. The Architecture
        rnn_model = Sequential()
        # rnn_model.add(normalizer) # Using the Normalization layer to standardize the datapoints during the forward pass
        # Input len(train) (input_shape=(?,?))
        rnn_model.add(LSTM(units=30, activation='tanh', return_sequences=True,
                           input_shape=(X_train_seq.shape[-2],X_train_seq.shape[-1])))
        # 2nd layer
        rnn_model.add(LSTM(units=20, activation='tanh', return_sequences=True))
        # 3rd layer
        rnn_model.add(LSTM(units=10, activation='relu'))
        # dense layer
        rnn_model.add(Dense(10, activation = 'relu')) ## add 1 or more 'relu' layers
        rnn_model.add(layers.Dropout(0.3)) ## if RNN model over-fit
        rnn_model.add(Dense(n_pred, activation = 'linear'))

        # 2. Compiling with 'rmsprop' rather than 'adam' (recommended)
        optimizer = RMSprop(learning_rate=learning_rate,
                            rho=0.9, momentum=0.0, epsilon=1e-07, centered=False)
        rnn_model.compile(loss='mse',
                      optimizer= optimizer, # optimizer='rmsprop'    <- adapt learning rate
                         metrics='mape')  # Recommended optimizer for RNNs

        # 3. Training
        train_rnn_model(model=rnn_model, X_train=X_train_seq, y_train=y_train_seq, X_val=X_val_seq, y_val=y_val_seq, epochs=200, patience=7)

        # 4. Evaluating
        # The prediction (one per sequence/city)
        y_pred = rnn_model.predict(X_test_seq)
        # Distribution of the predictions
        if return_test == True:
            return pd.DataFrame(y_pred).mean(), y_test
        return pd.DataFrame(y_pred).mean()

        # y = country_indicator['total_deaths']
