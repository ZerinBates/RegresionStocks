import tensorflow as tf
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
MONEYS=100000.00
df = DataReader.get_data_yahoo(symbols='CPSS', start=datetime(2012, 1, 1), end=datetime(2016, 12, 31))
df1 = DataReader.get_data_yahoo(symbols='CPSS', start=datetime(2017, 1, 1), end=datetime(2018, 12, 31))
df3 = DataReader.get_data_yahoo(symbols='ABDC', start=datetime(2012, 1, 1), end=datetime(2016, 12, 31))
df4 = DataReader.get_data_yahoo(symbols='ABDC', start=datetime(2017, 1, 1), end=datetime(2018, 12, 31))
df5= DataReader.get_data_yahoo(symbols='NEPT', start=datetime(2012, 1, 1), end=datetime(2016, 12, 31))
df6 = DataReader.get_data_yahoo(symbols='NEPT', start=datetime(2017, 1, 1), end=datetime(2018, 12, 31))
df7= DataReader.get_data_yahoo(symbols='AMZN', start=datetime(2012, 1, 1), end=datetime(2016, 12, 31))
df8 = DataReader.get_data_yahoo(symbols='AMZN', start=datetime(2017, 1, 1), end=datetime(2018, 12, 31))
df9= DataReader.get_data_yahoo(symbols='CHTR', start=datetime(2012, 1, 1), end=datetime(2016, 12, 31))
df10 = DataReader.get_data_yahoo(symbols='CHTR', start=datetime(2017, 1, 1), end=datetime(2018, 12, 31))
df11 = DataReader.get_data_yahoo(symbols='TYPE', start=datetime(2012, 1, 1), end=datetime(2016, 12, 31))
df12 = DataReader.get_data_yahoo(symbols='TYPE', start=datetime(2017, 1, 1), end=datetime(2018, 12, 31))
df13 = DataReader.get_data_yahoo(symbols='UEIC', start=datetime(2012, 1, 1), end=datetime(2016, 12, 31))
df14 = DataReader.get_data_yahoo(symbols='UEIC', start=datetime(2017, 1, 1), end=datetime(2018, 12, 31))
df15 = DataReader.get_data_yahoo(symbols='FWP', start=datetime(2012, 1, 1), end=datetime(2016, 12, 31))
df16 = DataReader.get_data_yahoo(symbols='FWP', start=datetime(2017, 1, 1), end=datetime(2018, 12, 31))
df17 = DataReader.get_data_yahoo(symbols='DMLP', start=datetime(2012, 1, 1), end=datetime(2016, 12, 31))
df18 = DataReader.get_data_yahoo(symbols='DMLP', start=datetime(2017, 1, 1), end=datetime(2018, 12, 31))
df19 = DataReader.get_data_yahoo(symbols='FKO', start=datetime(2012, 1, 1), end=datetime(2016, 12, 31))
df20 = DataReader.get_data_yahoo(symbols='FKO', start=datetime(2017, 1, 1), end=datetime(2018, 12, 31))
df21 = DataReader.get_data_yahoo(symbols='KBSF', start=datetime(2012, 1, 1), end=datetime(2016, 12, 31))
df22 = DataReader.get_data_yahoo(symbols='KBSF', start=datetime(2017, 1, 1), end=datetime(2018, 12, 31))
df23= DataReader.get_data_yahoo(symbols='SYMC', start=datetime(2012, 1, 1), end=datetime(2016, 12, 31))
df24 = DataReader.get_data_yahoo(symbols='SYMC', start=datetime(2017, 1, 1), end=datetime(2018, 12, 31))
df25= DataReader.get_data_yahoo(symbols='HWBK', start=datetime(2012, 1, 1), end=datetime(2016, 12, 31))
df26 = DataReader.get_data_yahoo(symbols='HWBK', start=datetime(2017, 1, 1), end=datetime(2018, 12, 31))
df27= DataReader.get_data_yahoo(symbols='VBIV', start=datetime(2012, 1, 1), end=datetime(2016, 12, 31))
df28 = DataReader.get_data_yahoo(symbols='VBIV', start=datetime(2017, 1, 1), end=datetime(2018, 12, 31))
df29 = DataReader.get_data_yahoo(symbols='RAND', start=datetime(2012, 1, 1), end=datetime(2016, 12, 31))
df30 = DataReader.get_data_yahoo(symbols='RAND', start=datetime(2017, 1, 1), end=datetime(2018, 12, 31))
df31 = DataReader.get_data_yahoo(symbols='LAMR', start=datetime(2012, 1, 1), end=datetime(2016, 12, 31))
df32 = DataReader.get_data_yahoo(symbols='LAMR', start=datetime(2017, 1, 1), end=datetime(2018, 12, 31))
df33 = DataReader.get_data_yahoo(symbols='PRQR', start=datetime(2012, 1, 1), end=datetime(2016, 12, 31))
df34 = DataReader.get_data_yahoo(symbols='PRQR', start=datetime(2017, 1, 1), end=datetime(2018, 12, 31))
df35 = DataReader.get_data_yahoo(symbols='FDX', start=datetime(2012, 1, 1), end=datetime(2016, 12, 31))
df36 = DataReader.get_data_yahoo(symbols='FDX', start=datetime(2017, 1, 1), end=datetime(2018, 12, 31))
new=[df1,df4,df6,df8,df10,df12,df14,df16,df18,df20,df22,df24,df26,df28,df30,df32,df34,df36]
old=[df,df3,df5,df7,df9,df11,df13,df15,df17,df19,df21,df23,df25,df27,df29,df31,df33,df35]


#print(df.index)
#df.to_cvs('d:/temp/aapl_data.csv')
# 0. Get the Data and simple sorting and check NaN
#df = pd.read_csv('Stocks/NCLH.csv',delimiter=',',usecols=['Date','Open','High','Low','Close'])
df1['Date']=pd.to_datetime(df1.index)
df1['Mean']= (df1.High+df1.Low)/2.0
df['Date'] = pd.to_datetime(df.index)
df['Mean'] = (df.High + df.Low )/2.0



#from statsmodels.tsa.seasonal import seasonal_decompose
#decomposition = seasonal_decompose(df.Mean.values, freq=365,two_sided=False) 
#trace1 = go.Scatter(
#    x = df.Date,y = decomposition.trend,
#    name = 'Trend',mode='lines'
#)
#trace2 = go.Scatter(
#    x = df.Date,y = decomposition.seasonal,
#    name = 'Seasonal',mode='lines'
#)
#algo=decomposition.seasonal[502:]
##print (df1)
def testBlank(stock):
    cur=MONEYS
    stocks=0
    x=0
    j=0
    for i in stock['Date']:
        x=stock[stock.Date==i]
        if(j>2):
                for k in range (0,5):
                
                    if(cur>x['Close'][0]):
                    
                        cur=cur-x['Close'][0]
                        stocks+=1
                    #print(cur)


          
        j+=1
    cur+=stocks*x['Close'][0]
    return cur
def testSeasonal(stock,algorithm):
    cur=MONEYS
    stocks=0
    j=0
    x=0
    w=0
    for i in stock['Date']:
        w+=1
        x=stock[stock.Date==i]
        if(j>2):
            if(algorithm[j]>algorithm[j-30]):
                for k in range (0,5):
                
                    if(cur>x['Close'][0]):
                       # print (x['Close'][0])
                        cur=cur-x['Close'][0]
                        stocks+=1
                    #print(cur)
            elif(algorithm[j]<algorithm[j-30]):
                for k in range (0,5):
                    if(stocks>0):
                        cur=cur+x['Close'][0]
                        stocks-=1
                    #print(cur)

          
        j+=1
    cur+=stocks*x['Close'][0]
    print(w)
    return cur



def testAr(stockNew,stockOld):
    from statsmodels.tsa.seasonal import seasonal_decompose
    window_size = 100
    run_avg_predictions = []
    running_mean = 0.0
    run_avg_predictions.append(running_mean)
    decay = 0.8
    Mean_list = list(stockOld.Mean)
    window_size = 50
    N = len(Mean_list)
    std_avg_predictions = list(Mean_list[:window_size])

    for pred_idx in range(1,N):
        running_mean = running_mean*decay + (1.0-decay)*Mean_list[pred_idx-1]
        run_avg_predictions.append(running_mean)

    from statsmodels.tsa.ar_model import AR
    window_size = 50
    ar_list = list(Mean_list[:window_size])
    for pred_idx in range(window_size,N):
        current_window = Mean_list[pred_idx-window_size:pred_idx]
        model = AR(current_window)
        model_fit = model.fit(49)
        
        current_predict = model_fit.predict(49,49)[0]
        ar_list.append(current_predict)
    cur=MONEYS
    stocks=0
    x=0
    window_size = 100
    run_avg_predictions = []
    #running_mean = 0.0
    run_avg_predictions.append(running_mean)
    decay = 0.8
    Mean_list = list(stockNew.Mean)
    window_size = 50
    N = len(Mean_list)
    std_avg_predictions = list(Mean_list[:window_size])
    t=1000
    stocks=0
    cur=MONEYS
    x=0
    for pred_idx in range(1,N):
        running_mean = running_mean*decay + (1.0-decay)*Mean_list[pred_idx-1]
        run_avg_predictions.append(running_mean)

    for pred_idx in range(window_size,N):
        current_window = Mean_list[pred_idx-window_size:pred_idx]
        model = AR(current_window)
        model_fit = model.fit(49)
        current_predict = model_fit.predict(49,49)[0]
        if(current_predict>t):
            for k in range (0,5):
                if(cur>Mean_list[pred_idx]):
                    cur=cur-Mean_list[pred_idx]
                    stocks+=1
        if(current_predict<t):
            for k in range(0,5):
                if(stocks>0):
                    cur=cur+Mean_list[pred_idx]


        ar_list.append(current_predict)
        t=current_predict
        x=Mean_list[pred_idx]

   
    cur+=stocks*x
    return cur

for i in range(0, 17):
    new[i]['Date']=pd.to_datetime(new[i].index)
    new[i]['Mean']= (new[i].High+new[i].Low)/2.0
    old[i]['Date'] = pd.to_datetime(old[i].index)
    old[i]['Mean'] = (old[i].High + old[i].Low )/2.0
    from statsmodels.tsa.seasonal import seasonal_decompose
    decomposition = seasonal_decompose(df.Mean.values, freq=365,two_sided=False) 
    decomposition = seasonal_decompose(old[i].Mean.values, freq=365,two_sided=False) 
    trace2 = go.Scatter(
        x = old[i].Date,y = decomposition.seasonal,
        name = 'Seasonal',mode='lines'
        )
    algo=decomposition.seasonal[-502:]
    
    print("Ar test      : "+str(testAr(new[i],old[i])))
    print("Seasonal test: "+str(testSeasonal(new[i],algo)))
    print("Monkey test: "+str(testBlank(new[i])))
