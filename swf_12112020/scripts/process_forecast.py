######################
#### Dependencies ####
######################
import os
import pandas as pd
import numpy as np
import sys
sys.path.append('../')

from utils import forecast_ingestion
###################
#### Init path ####
###################
main_dir = '../data/raw/forecasts/'

#####################################
#### Process dir and concat data ####
#####################################
forecast = forecast_ingestion.get_forecast(main_dir)

###################
#### Save file ####
###################
forecast.to_csv('../data/processed/last_forecast.csv',index=False)