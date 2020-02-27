import pandas as pd
import numpy as np
from pandas import read_csv
from pandas import read_csv

from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.metrics import mean_squared_error

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
from numpy.random import seed

#from data_preperation import * 
import tensorflow as tf

seed(6)
tf.random.set_seed(6)

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
    X = get_past_n_steps(df, steps_in)
    X2 = np.array(X)
    X3 = X2.reshape(-1, steps_in, X2.shape[1]//(steps_in))
    y_from_X = X3[steps_in:-2*steps_in,:steps_out,:3]
    return X3[:y_from_X.shape[0]], y_from_X

def scale_data_RNN(steps_in, steps_out, df, test_size = 0.2, speed_only = False):
    X, y = create_data_RNN(steps_in, steps_out, df)
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
    
    X_train_final_scaled = X_train_scaled.transpose(0,1,2)
    X_test_final_scaled = X_test_scaled.transpose(0,1,2)
    y_train_final_scaled = y_train_scaled.transpose(0,1,2)
    y_test_final_scaled = y_test_scaled.transpose(0,1,2)

    if speed_only:
        y_train_final_scaled = y_train_scaled[:,:,0]
        y_test_final_scaled = y_test_scaled[:,:,0]
    
    return X_train_final_scaled, X_test_final_scaled, y_train_final_scaled, y_test_final_scaled, X_train, X_test, y_train, y_test, scaler_X, scaler_y


def create_model(X_train, steps_in = 48, steps_out = 48, lr = 0.001, drop_out = 0, cell1size = 64, speed_only = False):
    n_features = X_train.shape[2]
    model = Sequential()
    model.add(LSTM(cell1size, activation='relu', input_shape=(steps_in, n_features)))
    model.add(Dropout(0.5))
    model.add(RepeatVector(steps_out))
    model.add(LSTM(cell1size, activation='relu', return_sequences=True))
    model.add(Dropout(0.5))
    if speed_only:
        model.add(TimeDistributed(Dense(1)))
    else:
        model.add(TimeDistributed(Dense(3)))#, activation='softmax')))
    opt = Adam(learning_rate= lr)
    model.compile(optimizer=opt, loss='mse')
    return model

def get_baseline_y_test(X_test, steps_out, speed_only):
    y_baseline = []
    for i in range(len(X_test)):
        s = X_test[i,-1,0]
        c = X_test[i,-1,1]
        sin = X_test[i,-1,2]
        if speed_only:
            y_baseline.append([s for i in range(steps_out)])
        else:
            y_baseline.append([[s, c, sin] for i in range(steps_out)])
    return np.array(y_baseline)

def baseline_loss(X_test, y_test, steps_out, speed_only):
    y_baseline = get_baseline_y_test(X_test, steps_out, speed_only)
    print(y_test.shape)
    print(y_baseline.shape)

    if speed_only:
        print("Baseline MSE speed: ", mean_squared_error(y_test.reshape(y_test.shape[0]*steps_out), y_baseline.reshape(y_test.shape[0]*steps_out)))

    else:
        print("Baseline MSE speed: ", mean_squared_error(y_test[:,:,0].reshape(y_test.shape[0]*steps_out), y_baseline[:,:,0].reshape(y_test.shape[0]*steps_out)))
        print("Baseline MSE cos: ", mean_squared_error(y_test[:,:,1].reshape(y_test.shape[0]*steps_out), y_baseline[:,:,1].reshape(y_test.shape[0]*steps_out)))
        print("Baseline MSE sin: ", mean_squared_error(y_test[:,:,2].reshape(y_test.shape[0]*steps_out), y_baseline[:,:,2].reshape(y_test.shape[0]*steps_out)))









