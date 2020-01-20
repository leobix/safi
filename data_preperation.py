import pandas as pd
import numpy as np
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from datetime import date, datetime, timedelta

#STEP 1: Read and Process Masurement Data
# Read 1 file per semester
def get_data():
    df0 = pd.read_csv("2015_S2.csv", sep = ";")
    df1 = pd.read_csv("2016_S1.csv", sep = ";")
    df2 = pd.read_csv("2016_S2.csv", sep = ";")
    df3 = pd.read_csv("2017_S1.csv", sep = ";")
    df4 = pd.read_csv("2017_S2.csv", sep = ";")
    df5 = pd.read_csv("2018_S1.csv", sep = ";")
    df6 = pd.read_csv("2018_S2.csv", sep = ";")
    df7 = pd.read_csv("2019_S1.csv", sep = ";")
    df8 = pd.read_csv("2019_S2.csv", sep = ";")

    data0 = pd.concat([df0, df1, df2, df3, df4, df5, df6, df7, df8], ignore_index=True)
    print('read csv semester csv files from 2015s2 to 2019s2')
    return data0


# Functions
# Convert scenario to one_hot
def scenario_one_hot(data, one_hot=False):
    # extract numeric data from scenario 'S1' to '1'
    data['scenario_num'] = (data['scenario'].str.extract('(\d+)')).astype(int)
    data.drop(['scenario'], axis=1, inplace=True)

    # add one-hot encoding to scenario:
    if one_hot:
        scenario = pd.get_dummies(data['scenario_num'], prefix='scenario', dummy_na=True)
        data1 = pd.concat([data, scenario], axis=1)
        return data1
    return data


# Make cyclical data into continuous data using cos & sin
def smooth_wind_dir(data):
    data['cos_wind_dir'] = np.cos(2 * np.pi * data['wind_dir'] / 360)
    data['sin_wind_dir'] = np.sin(2 * np.pi * data['wind_dir'] / 360)
    print('smooth wind direction')
    return data


def smooth_hour(data):
    # split '00h00' to two columns of numeric values
    hour = data['hour'].str.split(pat='h', expand=True)
    hour = hour.apply(pd.to_numeric, errors='coerce')

    # calculate minutes passed since 00h00
    hour['minutes'] = 60 * hour[0] + hour[1]
    hour['cos_hour'] = np.cos(2 * np.pi * hour['minutes'] / (60 * 24))
    hour['sin_hour'] = np.sin(2 * np.pi * hour['minutes'] / (60 * 24))

    # concat and update dataframe
    data = pd.concat([data, hour[['cos_hour', 'sin_hour']]], axis=1)
    print('smooth hour')
    return data


# Smooth date
def smooth_day(data):
    # Convert day & hour to date-time format
    data['datetime'] = data['day'].str.cat(data['hour'], sep=' ')
    data['datetime'] = pd.to_datetime(data['datetime'], format='%d/%m/%Y %Hh%M')
    data['day'] = pd.to_datetime(data['day'], format='%d/%m/%Y')
    data['hour'] = data['hour'].str.extract('(\d+)')
    data['hour'] = pd.to_numeric(data['hour'])

    # Calculate time delta since 1st entry
    data['day_delta'] = pd.to_numeric(data['day'] - data['day'][0])
    data['cos_day'] = np.cos(2 * np.pi * data['day_delta'] / (365))
    data['sin_day'] = np.sin(2 * np.pi * data['day_delta'] / (365))
    data.drop(['day_delta', 'day'], axis=1, inplace=True)

    print('smooth day')
    return data


# Generate new features:
# Generate daily features: daily min, max
def generate_daily(df):
    # group data into daily batches
    grouped = df.resample('D')
    min_speed = []
    max_speed = []
    min_hour = []
    max_hour = []

    for datetime, group in grouped:
        # find daily min & max
        s1 = group['speed'].min()
        s2 = group['speed'].max()
        # find the time of min & max speed
        h1 = group.loc[group['speed'] == s1]['hour'].values[0]
        h2 = group.loc[group['speed'] == s2]['hour'].values[0]

        min_speed.append(s1)
        max_speed.append(s2)
        min_hour.append(h1)
        max_hour.append(h2)

    # output new features as a dataframe
    start = df.index[0].date()
    end = df.index[-1].date()
    date_range = pd.date_range(start, end, freq='D')
    daily = pd.concat([pd.Series(min_speed), pd.Series(min_hour), pd.Series(max_speed), pd.Series(max_hour)], axis=1,
                      keys=['daily_min_speed', 'daily_min_hour', 'daily_max_speed', 'daily_max_hour'])
    daily.set_index(date_range, inplace=True)

    # #merge new features into dataframe: match with date
    df_out = pd.merge(df, daily, how='outer', left_index=True, right_index=True)
    # fill NaN values with same daily values
    df_out = df_out.fillna(method='ffill')

    print('generate daily features: %s' % (daily.columns.to_list()))
    return df_out


# Categorical features
def generate_season(df):
    df['season'] = 0
    df['month'] = df.index.month
    df.loc[df['month'].isin([12, 1, 2]), 'season'] = 1
    df.loc[df['month'].isin([3, 4, 5]), 'season'] = 2
    df.loc[df['month'].isin([6, 7, 8]), 'season'] = 3
    df.loc[df['month'].isin([9, 10, 11]), 'season'] = 4
    df.drop(['month'], axis=1, inplace=True)
    print('generate seasonality categorical feature')
    return df


def generate_day_night(df):
    df['day'] = 0
    df['night'] = 0
    df.loc[df['hour'].isin([8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]), 'day'] = 1
    df.loc[df['hour'].isin([0, 1, 2, 3, 4, 5, 6, 7, 19, 20, 21, 22, 23]), 'night'] = 1
    print('generate day/night categorical feature')
    return df


# Function to prepare data using above functions
def prepare_data(one_hot=False):
    # Interpolate missing values
    data0 = get_data()
    data = data0.interpolate()
    data = data.fillna(method='ffill')

    # scenario to one-hot encoding
    data = scenario_one_hot(data, one_hot)

    # smooth wind_dir, hour, and day using cos & sin function
    data = smooth_wind_dir(data)
    data = smooth_hour(data)
    data = smooth_day(data)
    data.drop(['details'], axis=1, inplace=True)
    data.index = data['datetime']
    data = data.interpolate()

    # averaging 15min data to hourly
    data = data.resample('H').mean()
    data = data.round({'scenario_num': 0})

    # generate daily max & min wind speed features
    data = generate_daily(data)

    # generate seasonal, day/night categorical features
    data = generate_season(data)
    data = generate_day_night(data)

    return data


#STEP 2: Merging with Forecast Data
#get forecast data
def get_forecast_data():
    f00 = pd.read_csv("Data/forecast_00.csv")
    f12 = pd.read_csv("Data/forecast_12.csv")
    f24 = pd.read_csv("Data/forecast_24.csv")
    f36 = pd.read_csv("Data/forecast_36.csv")
    f48 = pd.read_csv("Data/forecast_48.csv")
    return f00, f12, f24, f36, f48

#Functions to process forecast data
#convert to datetime index
def convert_datetime(df):
    df['datetime'] = pd.to_datetime(df['date'], format='%m/%d/%y %H:%M')
    df.drop(['date', 'cycle'], axis=1, inplace=True)
    df.set_index('datetime', inplace=True)
    return df

#rename columns
def rename_cols(df):
    df_out = df.rename(columns={"direction (ｰ)": "wind_dir", "vitesse (m/s)": "speed", "temperature (ｰC)": "temp", "rayonnement (W/m2)": "radiation","precip (mm/h)":"precip"})
    return df_out

#Function to merge data with forecast data
def prepare_data_with_forecast(data):
    #get prepared measurement data
    data_merge=data.copy()
    #get forecast data
    f00, f12, f24, f36, f48 = get_forecast_data()
    name_str = ['f00', 'f12', 'f24', 'f36', 'f48']
    i = 0
    for df in [f00, f12, f24, f36, f48]:
        df_temp = convert_datetime(df)
        df_temp = rename_cols(df_temp)
        data_merge = data_merge.join(df_temp, how='left', rsuffix='_'+name_str[i])
        i+=1
    print('merged with forecast data '+ str(name_str))
    return data_merge
#data=prepare_data(one_hot=False)
#data_merge = prepare_data_with_forecast(data)
