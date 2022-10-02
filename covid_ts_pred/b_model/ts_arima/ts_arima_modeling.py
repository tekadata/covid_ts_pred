## canonical import of libraries
from tkinter.tix import Tree
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_profiling


Tree

data = pd.read_csv('/root/code/bktan69/data-challenges/')




## Baseline (simplest model)
# 1 feature only: X[t] = y[t-1] to predict the previous value!
y = data.value
data.plot()


# ensure stationnarity with Dick Fuller test
from statsmodels.tsa.stattools import adfuller
print('p-value: ', adfuller(y)[1])

# find the minimum difference to make it stationnary
# 0 diff
print('p-value: ', adfuller(y)[1])

# First diff
print('p-value: ', adfuller(y.diff(1).fillna(0))[1])

# Second order diff
print('p-value: ', adfuller(y.diff(1).diff(1).fillna(0))[1])



# plot differencing
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Original Series
fig, axes = plt.subplots(3, 2, figsize=(13,10))
axes[0, 0].plot(y); axes[0, 0].set_title('Original Series')
plot_acf(y, auto_ylims=True,ax=axes[0, 1])

# 1st Differencing
axes[1, 0].plot(y.diff()); axes[1, 0].set_title('1st Order Differencing')
plot_acf(y.diff().dropna(), auto_ylims=True,ax=axes[1, 1])

# 2nd Differencing
axes[2, 0].plot(y.diff().diff()); axes[2, 0].set_title('2nd Order Differencing')
plot_acf(y.diff().diff().dropna(), auto_ylims=True, ax=axes[2, 1])

plt.tight_layout()


# keep only one diff order
y_diff = y.diff().dropna()


# determine q
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plot_acf(y_diff, auto_ylims=True)

# determine p
plot_pacf(y_diff, method='ywm', auto_ylims=True, c='r')


## build the model
from statsmodels.tsa.arima.model import ARIMA

# 1. initialize the model
arima = ARIMA(y, order=(1, 1, 2), trend='t')

# 2. fit the models
arima = arima.fit()

arima.summary()


## evaluate the model performance
from statsmodels.graphics.tsaplots import plot_predict

# Actual vs Fitted
plot_predict(arima, start=1, dynamic=False)
plt.ylim((.9 * y.min(), 1.1 * y.max()))


# Try to use `dynamic=True` to plot a prection of the _last 15 values_ in a situation where the model only have _access to data up to 85_. That is to say, the model:
# * predicts 86 based on true [1...85]
# * then predicts 87 based on [1...85] _plus_ it's previouly predicted value for 86
# * etc...iteratively until 100
plot_predict(arima, start=86, end=100, dynamic=True)
plt.legend(loc='upper left')




## out-of-sample forecasts real future
y_train = y[:85]
y_test = y[85:]
arima = ARIMA(y_train, order=(1, 1, 2), trend='t').fit()


# we are now in the 85 future steps
forecast_results = arima.get_forecast(15, alpha=0.05)

forecast = forecast_results.predicted_mean.reset_index(drop=True)
# forecast = arima.forecast(15, alpha=0.05)
conf_int = pd.DataFrame(forecast_results.conf_int().values, columns=['low', 'high'])

#plot forecasted values
plt.plot(forecast, c='orange')
plt.plot(conf_int['low'], label='low', c='grey', ls='--')
plt.plot(conf_int['high'], label='high', c='grey', ls='--')
plt.legend()


# plot the 85 previous data points
forecast.index = y_test.index
conf_int.index = y_test.index

plt.figure(figsize=(15,5))
plt.plot(forecast, c='orange')
plt.plot(conf_int['low'], label='low', c='orange', ls='--')
plt.plot(conf_int['high'], label='high', c='orange', ls='--')
plt.plot(y_train, c='blue')
plt.plot(y_test, c='blue')
plt.legend()
plt.fill_between(y_test.index, conf_int['low'], conf_int['high'], color='k', alpha=.15);



# plot the residuals
residuals = pd.DataFrame(arima.resid)
residuals.plot(title="Residuals")


residuals.plot(kind='kde', title='Residual density')


# cross validated performance metrics

import numpy as np
from statsmodels.tsa.stattools import acf
def forecast_accuracy(y_pred: pd.Series, y_true: pd.Series) -> float:

    mape = np.mean(np.abs(y_pred - y_true)/np.abs(y_true))  # Mean Absolute Percentage Error
    me = np.mean(y_pred - y_true)             # ME
    mae = np.mean(np.abs(y_pred - y_true))    # MAE
    mpe = np.mean((y_pred - y_true)/y_true)   # MPE
    rmse = np.mean((y_pred - y_true)**2)**.5  # RMSE
    corr = np.corrcoef(y_pred, y_true)[0,1]   # Correlation between the Actual and the Forecast
    mins = np.amin(np.hstack([y_pred.values.reshape(-1,1), y_true.values.reshape(-1,1)]), axis=1)
    maxs = np.amax(np.hstack([y_pred.values.reshape(-1,1), y_true.values.reshape(-1,1)]), axis=1)
    minmax = 1 - np.mean(mins/maxs)             # minmax
    acf1 = acf(y_pred-y_true, fft=False)[1]                      # Lag 1 Autocorrelation of Error
    return({'mape':mape, 'me':me, 'mae': mae,
            'mpe': mpe, 'rmse':rmse, 'acf1':acf1,
            'corr':corr, 'minmax':minmax})


# ARIMA hyperparameters and impact on forecast performance
forecast_accuracy(forecast, y_test)




## GridSearch


import pmdarima as pm

model = pm.auto_arima(y_train,
                      start_p=0, max_p=3,
                      start_q=0, max_q=3,
                      d=None,           # let model determine 'd'
                      test='adf',       # using adftest to find optimal 'd'
                      trace=True, error_action='ignore',  suppress_warnings=True)
print(model.summary())



## ARIMA cross validation and GridSearch

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score
import itertools
from statsmodels.tsa.statespace.sarimax import SARIMAX


range_p = [0, 1, 2]
range_d = [1, 2]
range_q = [0, 1, 2]
grid = itertools.product(range_p, range_d, range_q)
orders = []
r2s = []
aics = []
fold_idxs = []
y = y.astype('float32')
for (p,d,q) in grid:
    order = (p,d,q)
    folds = TimeSeriesSplit(n_splits=3)
    for fold_idx, (train_idx, test_idx) in enumerate(folds.split(y)):
        fold_idxs.append(fold_idx)
        y_train = y[train_idx]
        y_test = y[test_idx]
# Using SARIMAX without seasonality nor exogeneous variables is equivalent to using ARIMA
# SARIMAX's fit method comes with a maxiter keyword, useful to avoid warnings
        model = SARIMAX(y_train, order=order).fit(maxiter=75, disp=False)
#         model = ARIMA(y_train, order=order).fit()
        y_pred = model.forecast(len(y_test))
        r2s.append(r2_score(y_test, y_pred))
        orders.append(order)
        aics.append(model.aic)


df = pd.DataFrame(list(zip(fold_idxs, orders, aics, r2s)),
                   columns =['Fold', '(p,d,q)', 'AIC', 'R2'])




# AIC according to hyperparameters

df.sort_values('AIC').groupby('(p,d,q)').mean()['AIC'].sort_values()
