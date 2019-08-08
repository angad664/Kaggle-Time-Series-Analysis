#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 17:41:46 2019

@author: angadsingh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import rcParams
rcParams['figure.figsize'] = 10, 6

dataset = pd.read_csv('AirPassengers.csv')
dataset['Month'] = pd.to_datetime(dataset['Month'], infer_datetime_format = True)
IndexedDataset = dataset.set_index(['Month'])

from datetime import datetime
IndexedDataset.head(5)

#plot graph
plt.xlabel("Date")
plt.ylabel("Number of Air Passenger")
plt.plot(IndexedDataset)

# rolling statistics - to check the stationarity of the data
rolmean = IndexedDataset.rolling(window = 12).mean() # window = 12 means months
rolstd = IndexedDataset.rolling(window= 12).std()
print(rolmean, rolstd)

# plot rolling statistics
original = plt.plot(IndexedDataset, color ='blue', label = 'Original')
mean = plt.plot(rolmean, color='red', label = 'Rolling Mean')
std = plt.plot(rolstd, color = 'black', label = 'Rolling Standard Deviation')
plt.legend(loc ='best')
plt.title('Rolling mean and Standard Deviation')
plt.show(block = False)

# Perform Dickey fuller test
from statsmodels.tsa.stattools import adfuller
print('Result of Dickey-fuller Test')
dftest = adfuller(IndexedDataset['#Passengers'], autolag ='AIC')

dfoutput = pd.Series(dftest[0:4], index = ['Test Statistics', 'p-value', '#lags used', 'Number of Observation used'])
for key,value in dftest[4].items():
    dfoutput['Critical Value (%s)' %key] = value

print(dfoutput)
# from p value we know data is not stationary if stationary p should be around 0.5


# estimating trend
IndexedDataset_logscale = np.log(IndexedDataset)
plt.plot(IndexedDataset_logscale)

# calculate moving average
movingAverage = IndexedDataset_logscale.rolling(window = 12).mean()
movingSTD = IndexedDataset_logscale.rolling(window=12).std()
plt.plot(IndexedDataset_logscale)
plt.plot(movingAverage, color='red')

#difference
df = IndexedDataset_logscale - movingAverage
df.head(12)

#remove nan values
df.dropna(inplace = True)
df.head(10)

# ADCF test
from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    #determining rolling statistics
    movingAverage = timeseries.rolling(window=12).mean()
    movingSTD = timeseries.rolling(window=12).std()
    #plot rolling stats
    original = plt.plot(timeseries, color='blue', label = 'Original')
    mean = plt.plot(movingAverage, color='red', label = 'Rolling mean')
    std = plt.plot(movingSTD, color='black', label = 'Rolling STD')
    plt.legend(loc = 'best')
    plt.title('Rolling mean and Standard Deviation')
    plt.show(block = False)

# perform dickey fuller test
    from statsmodels.tsa.stattools import adfuller
    print('Result of Dickey-fuller Test')
    dftest = adfuller(IndexedDataset['#Passengers'], autolag ='AIC')

    dfoutput = pd.Series(dftest[0:4], index = ['Test Statistics', 'p-value', '#lags used', 'Number of Observation used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)' %key] = value

    print(dfoutput)

# trend present inside timeseries
exponentialDecayWeightAverage = IndexedDataset_logscale.ewm(halflife=12, min_periods= 0, adjust= True).mean()
plt.plot(IndexedDataset_logscale)
plt.plot(exponentialDecayWeightAverage, color='red')

# dataset_logscale - movingexponentialDecayWeightAverage
dv = IndexedDataset_logscale - exponentialDecayWeightAverage
test_stationarity(dv)   # p value = 0.005 means our data is stationary

# to shift the values
datasetlogDiffShift = IndexedDataset_logscale - IndexedDataset_logscale.shift()
plt.plot(datasetlogDiffShift)

datasetlogDiffShift.dropna(inplace = True)
test_stationarity(datasetlogDiffShift)  # rolling mean and rolling std is quite flat so, stationary

from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(IndexedDataset_logscale)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.subplot(411)
plt.plot(IndexedDataset_logscale, label = 'Original')
plt.legend(loc ='best')
plt.subplot(412)
plt.plot(trend, label = 'trend')
plt.legend(loc ='best')
plt.subplot(413)
plt.plot(seasonal, label = 'Seasonality')
plt.legend(loc ='best')
plt.subplot(414)
plt.plot(residual, label = 'Residuals')
plt.legend(loc ='best')
plt.tight_layout()


# check the noise/residual/irregularity is stationary or not
decomposedlogdata = residual
decomposedlogdata.dropna(inplace = True)
test_stationarity(decomposedlogdata)

#ACF(P) and PACF test (Q)
from statsmodels.tsa.stattools import acf, pacf
lag_acf = acf(datasetlogDiffShift, nlags = 20)
lag_pacf = pacf(datasetlogDiffShift, nlags = 20, method = 'ols')

#Plot ACF
plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y = 0, linestyle ='--', color = 'gray')
plt.axhline(y =-1.96/np.sqrt(len(datasetlogDiffShift)), linestyle='--', color = 'gray')
plt.axhline(y =1.96/np.sqrt(len(datasetlogDiffShift)), linestyle='--', color = 'gray')
plt.title('Autocorrelation Function')

#plot PACF
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y = 0, linestyle =' --', color = 'gray')
plt.axhline(y =-1.96/np.sqrt(len(datasetlogDiffShift)), linestyle='- -', color = 'gray')
plt.axhline(y =1.96/np.sqrt(len(datasetlogDiffShift)), linestyle='- -', color = 'gray')
plt.title('Partial Autocorrelation Function')


from statsmodels.tsa.arima_model import ARIMA
# AR model
model = ARIMA(IndexedDataset_logscale, order = (2,1,2))# p =2, d =1, q = 2
result_AR = model.fit(disp =-1)
plt.plot(datasetlogDiffShift)
plt.plot(result_AR.fittedvalues, color ='red')
plt.title('RSS: %.4f '% sum((result_AR.fittedvalues-datasetlogDiffShift["#Passengers"])**2)) # RSS residual sum of squares
print('Plotting AR Model')

#MA model
model = ARIMA(IndexedDataset_logscale, order = (2,1,0))
result_AR = model.fit(disp =-1)
plt.plot(datasetlogDiffShift)
plt.plot(result_AR.fittedvalues, color ='red')
plt.title('RSS: %.4f '% sum((result_AR.fittedvalues-datasetlogDiffShift["#Passengers"])**2))
print('Plotting AR Model')

# ARIMA
model = ARIMA(IndexedDataset_logscale, order = (2,1,2))
result_AR = model.fit(disp =-1)
plt.plot(datasetlogDiffShift)
plt.plot(result_AR.fittedvalues, color ='red')
plt.title('RSS: %.4f '% sum((result_AR.fittedvalues-datasetlogDiffShift["#Passengers"])**2))


prediction_ARIMA_diff = pd.Series(result_AR.fittedvalues, copy=True)
print(prediction_ARIMA_diff.head())

# DATA TRANSFORMATION
# convert to cumalative sum
prediction_ARIMA_diff_cumsum = prediction_ARIMA_diff.cumsum()
print(prediction_ARIMA_diff_cumsum)

#prediction for fitted values
prediction_ARIMA_log = pd.Series(IndexedDataset_logscale['#Passengers'].ix[0],index=IndexedDataset_logscale.index)
prediction_ARIMA_log = prediction_ARIMA_log.add(prediction_ARIMA_diff_cumsum, fill_value=0)
prediction_ARIMA_log.head()

# taking exponent so that it can come in original form
prediction_ARIMA = np.exp(prediction_ARIMA_log)
plt.plot(IndexedDataset)
plt.plot(prediction_ARIMA)

IndexedDataset_logscale

# if we want to predict for next ten years it would be 12months x 10 = 120
result_AR.plot_predict(1,264) # 144 rows +120   # for visualization
result_AR.forecast(steps=120)     # in array format
