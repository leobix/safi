######################
#### Dependencies ####
######################
import sys
sys.path.append('../')
import os
import pandas as pd
import numpy as np
import datetime 
from utils import measurement_ingestion

###################
#### Init days ####
###################
sys_date = sys.argv[1]
sys_year = pd.to_datetime(sys_date, format = '%d/%m/%Y').strftime(format='%y')
DAY1 = pd.to_datetime(sys_date, format = '%d/%m/%Y').strftime(format='%d-%m-%y')
DAY2 = (pd.to_datetime(sys_date, format = '%d/%m/%Y') - datetime.timedelta(days=1)).strftime(format='%d-%m-%y')
####################
#### Init paths ####
####################
main_dir = '//Sa-modat-cs-pr/plumair/SAFI/Entrees/' + '20' + sys_year + '/'
save_dir = '../data/raw/measurements/'

###################
#### Save file ####
###################
data = measurement_ingestion.get_one_day_measurement(DAY1,DAY2,main_dir)
data.to_csv(save_dir + 'measurement_' + DAY1 + '.csv',index=False)