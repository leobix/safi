import pandas as pd
import numpy as np

# smooth wind direction into cos and sin:
def smooth_wind_dir(df):
    df['cos_wind_dir'] = np.cos(2 * np.pi * df['wind_dir'] / 360)
    df['sin_wind_dir'] = np.sin(2 * np.pi * df['wind_dir'] / 360)
    df.drop(columns=['wind_dir'], inplace=True)
    return df

# keep the latest value per forecast date 
def keep_last_forecast (df):
    df.sort_values(by=['f_date','p_date'], inplace=True)
    df.drop_duplicates(subset = 'f_date', keep = 'last', inplace=True)
    df.set_index('f_date', inplace=True)
    df.drop(['p_date','f_period'], axis=1, inplace=True)
    return df