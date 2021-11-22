######################
#### Dependencies ####
######################
import os 
import numpy as np
import pandas as pd
import datetime

import sys
sys.path.append('../')
from utils.util_functions import *

# Define dir and tree
main_dir = '../data/raw/forecasts/'
sub_dir = os.listdir(main_dir)

# Date range to fetch : 5 months
end_date = pd.to_datetime(datetime.datetime.now())
start_date = pd.to_datetime(datetime.datetime.now() - datetime.timedelta(days=15))

# Concat files
forecast = pd.DataFrame()
for file_path in sub_dir:
    DAY = file_path[0:10]
    HOUR = file_path[11:13]
    file_path_date = pd.to_datetime(DAY.replace('_','-') + ' ' + HOUR + ':00:00')
    if (start_date <= file_path_date) & (file_path_date + datetime.timedelta(hours=8) <= end_date):
        forecast_part = get_one_forecast(DAY,HOUR)
        forecast_part['file_creation_date'] = file_path_date + datetime.timedelta(hours=8)
        forecast = forecast.append(forecast_part)
        del forecast_part
forecast = forecast.reset_index(drop=True)

# Change to dt 
forecast['f_date']= pd.to_datetime(forecast['f_date'],format='%Y-%m-%d %H:%M:%S')
forecast['p_date']= pd.to_datetime(forecast['p_date'],format='%Y-%m-%d %H:%M:%S')

# Save file
forecast.to_csv('../data/processed/last_forecast.csv',index=False)