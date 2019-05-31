#import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np,sys
import pandas as pd
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go

import matplotlib.pyplot as plt
from numpy import newaxis
import pandas_datareader as DataReader
from datetime import datetime
import fix_yahoo_finance as yf

df = DataReader.get_data_yahoo(symbols='AMZN', start=datetime(2000, 1, 1), end=datetime(2012, 1, 1))
#print(df.index)
#df.to_cvs('d:/temp/aapl_data.csv')
# 0. Get the Data and simple sorting and check NaN
#df = pd.read_csv('Stocks/NCLH.csv',delimiter=',',usecols=['Date','Open','High','Low','Close'])
df['Date'] = pd.to_datetime(df.index)
df['Mean'] = (df.High + df.Low )/2.0



from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(df.Mean.values, freq=365,two_sided=False) 
trace1 = go.Scatter(
    x = df.Date,y = decomposition.trend,
    name = 'Trend',mode='lines'
)
trace2 = go.Scatter(
    x = df.Date,y = decomposition.seasonal,
    name = 'Seasonal',mode='lines'
)
trace3 = go.Scatter(
    x = df.Date,y = decomposition.resid,
    name = 'Residual',mode='lines'
)
trace4 = go.Scatter(
    x = df.Date,y = df.Mean,
    name = 'Mean Stock Value',mode='lines'
)


# a. Standard Average of Window
Mean_list = list(df.Mean)
window_size = 50
N = len(Mean_list)
std_avg_predictions = list(Mean_list[:window_size])
for pred_idx in range(window_size,N):
    std_avg_predictions.append(np.mean(Mean_list[pred_idx-window_size:pred_idx]))

# b. EXP Average of Window
window_size = 100
run_avg_predictions = []
running_mean = 0.0
run_avg_predictions.append(running_mean)
decay = 0.8

for pred_idx in range(1,N):
    running_mean = running_mean*decay + (1.0-decay)*Mean_list[pred_idx-1]
    run_avg_predictions.append(running_mean)

trace5 = go.Scatter(
    x = df.Date,y = std_avg_predictions,
    name = 'Window AVG',mode='lines'
)
trace6 = go.Scatter(
    x = df.Date,y = run_avg_predictions,
    name = 'Moving AVG',mode='lines'
)

from statsmodels.tsa.ar_model import AR
window_size = 50
ar_list = list(Mean_list[:window_size])
for pred_idx in range(window_size,N):

    current_window = Mean_list[pred_idx-window_size:pred_idx]
    model = AR(current_window)
    model_fit = model.fit(49)
    current_predict = model_fit.predict(49,49)[0]
    ar_list.append(current_predict)

trace7 = go.Scatter(
    x = df.Date,y = ar_list,
    name = 'Auto Regression',mode='lines'
)
trace7
data = [trace1,trace2,trace3,trace4,trace5,trace6,trace7]
#data = [trace1,trace2]
plot(data)

