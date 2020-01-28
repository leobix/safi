import pandas as pd
import numpy as np
from pandas import read_csv
from matplotlib import pyplot
import matplotlib.pyplot as plt
from pandas.plotting import lag_plot
from pandas import read_csv
from matplotlib import pyplot
from pandas.plotting import autocorrelation_plot

from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error

from statsmodels.tsa.ar_model import AR
from sklearn.metrics import mean_squared_error

from statsmodels.tsa.arima_model import ARIMA

#Time
from datetime import date, datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import Dropout

from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam

from data_preperation import * 


data=prepare_data(one_hot=False)
data_merge = prepare_data_with_forecast(data)


#select useful columns 
df= data_merge[['speed', 'cos_wind_dir', 'sin_wind_dir', 'temp', 'radiation', 'precip',
       'scenario_num', 'cos_hour', 'sin_hour',
       'cos_day', 'sin_day', 'daily_min_speed', 'daily_min_hour',
       'daily_max_speed', 'daily_max_hour', 'season', 'day', 'night']]#,
       #'wind_dir_f00', 'speed_f00', 'cos_wind_dir_f00', 'sin_wind_dir_f00',
       #'wind_dir_f12', 'speed_f12', 'cos_wind_dir_f12', 'sin_wind_dir_f12',
       #'wind_dir_f24', 'speed_f24', 'cos_wind_dir_f24', 'sin_wind_dir_f24']]

def get_past_n_steps(df, steps_in):
    df_out = df.copy().add_suffix('_t-'+str(steps_in))
    for i in range(1, steps_in): 
        df_temp= df.shift(periods=-i, axis=0) #shift down 1 row 
        df_out=df_out.join(df_temp, how = 'inner', rsuffix='_t-'+str(steps_in-i))
    #reset datetime index
    df_out['datetime']=df_out.index.to_series()+timedelta(hours=steps_in) #shift n_steps
    df_out.set_index(pd.DatetimeIndex(df_out['datetime']), inplace=True) #reset index
    df_out.drop('datetime', axis=1, inplace=True)
    #unsure of this
    df_out.interpolate(inplace= True)
    df_out.fillna(method='ffill', inplace= True)
    df_out.dropna(inplace= True)
    return df_out 

def get_future_n_steps(df, steps_out):
    df_out = df.copy().add_suffix('_t')
    for i in range(1,steps_out): 
        df_temp= df.shift(periods=+i, axis=0) #shift up 1 row 
        df_out=df_out.join(df_temp, how = 'inner', rsuffix='_t+'+str(i))
    #reset datetime index
    df_out['datetime']=df_out.index.to_series()+timedelta(hours=steps_out) #shift n_steps
    df_out.set_index(pd.DatetimeIndex(df_out['datetime']), inplace=True) #reset index
    df_out.drop('datetime', axis=1, inplace=True)
    df_out.interpolate(inplace= True)
    df_out.fillna(method='ffill', inplace= True)
    df_out.dropna(inplace= True)
    return df_out     


def create_data_RNN(steps_in, steps_out, df):
    X, y = get_past_n_steps(df, steps_in), get_future_n_steps(df, steps_out)
    X2, y2 = np.array(X), np.array(y)
    X3, y3 = X2.reshape(-1, steps_in, X2.shape[1]//(steps_in)), y2.reshape(-1, steps_out, y2.shape[1]//(steps_out))
    y_from_X = X3[steps_in:,:,:3]
    return X3[:y_from_X.shape[0]], y_from_X




def scale_data_RNN(steps_in, steps_out, df, test_size = 0.2):
    X, y = create_data_RNN(steps_in, steps_out, df)
    #y = y[:,:,:3]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle = False)
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    
    X_train2 = X_train.reshape(X_train.shape[0]*X_train.shape[1], X_train.shape[2])
    X_test2 = X_test.reshape(X_test.shape[0]*X_test.shape[1], X_test.shape[2])
    y_train2 = y_train.reshape(y_train.shape[0]*y_train.shape[1], y_train.shape[2])
    y_test2 = y_test.reshape(y_test.shape[0]*y_test.shape[1], y_test.shape[2])
    
    X_train_scaled = scaler_X.fit_transform(X_train2)
    X_test_scaled = scaler_X.transform(X_test2)
    y_train_scaled = scaler_y.fit_transform(y_train2)
    y_test_scaled = scaler_y.transform(y_test2)
    
    X_train_scaled = X_train_scaled.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2])
    X_test_scaled = X_test_scaled.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2])
    y_train_scaled = y_train_scaled.reshape(y_train.shape[0], y_train.shape[1], y_train.shape[2])
    y_test_scaled = y_test_scaled.reshape(y_test.shape[0], y_test.shape[1], y_test.shape[2])
    
    X_train_final = X_train_scaled.transpose(0,1,2)
    X_test_final = X_test_scaled.transpose(0,1,2)
    y_train_final = y_train_scaled.transpose(0,1,2)
    y_test_final = y_test_scaled.transpose(0,1,2)
    
    #X_train_final = X_train_scaled.transpose(0,2,1)
    #X_test_final = X_test_scaled.transpose(0,2,1)
    #y_train_final = y_train_scaled.transpose(0,2,1)
    #y_test_final = y_test_scaled.transpose(0,2,1)
    
    return X_train_final, X_test_final, y_train_final, y_test_final, y_test, scaler_X, scaler_y


def create_model(X_train, steps_in = 48, steps_out = 48, lr = 1e-3, drop_out = 0, cell1size = 64):
    n_features = X_train.shape[2]
    model2 = Sequential()
    model2.add(LSTM(cell1size, activation='relu', input_shape=(steps_in, n_features)))
    model2.add(Dropout(0.5))
    model2.add(RepeatVector(steps_out))
    model2.add(LSTM(cell1size, activation='relu', return_sequences=True))
    model2.add(Dropout(0.5))
    model2.add(TimeDistributed(Dense(3)))#, activation='softmax')))
    opt = Adam(learning_rate=lr)
    model2.compile(optimizer=opt, loss='mse')
    return model2