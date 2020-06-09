This documentation serves the purpose to share current running codes and results, which can be turned into a beta version for testing phase. 

Individual functions are coded in separate python files: 
1. ```utils/data_preparation.py``` is the file containing functions to process measurements dataframe (data obtained by Plum air device) as well as forecast dataframe (official forecast from the weather agency). Specifically, it includes functions to: 
    - read raw CSV files for measurement(two per year from 2015S1 to 2020S1) and official government forecast data
    - clean data, fill missing values 
    - smooth wind angle into continuous data using cos and sin functions
    - extract some key features such as (time of the day, seasonality feature)  
    - concatenate everything  into one dataframe 
    
    
2. ```utils/data_process.py``` is the file reading the previous cleaned dataframes and outputs data ready to be used for prediction algorithms:
    - at each present time, enrich past $n$ time steps of measurement data
    - at each prediction time, enrich forecast data for the forecasted time
    
3. ```utils/utils_scenario.py``` contains miscellaneous functions
    - in particular from a wind speed and wind direction it outputs the scenario, and dangerous or not. This can be changed easily to reflect any change of policy defining scenarios.
    
4. ```main_XGB.py``` is the main function to be called to perform XGB regression and classification. You can amend the following arguments:
    - steps_in: number of past data for prediction, default = 48 
    - t_list: prediction time steps, default = [1,3,6,9,12,15,18,21,24,27,30,33,36,39,42,45,48]
    
5. ```main_OCT.py``` is the main function to be called to perform Optimal Trees for regression and classification. You can amend in particular the following arguments:
    - steps-in: number of past data for prediction, default = 48 (hours of data)
    - steps-out: what specific hour do you want to build a model for.

Note: this function can take a long time to run in Jupyter notebook. We have written a python file called 'main_XGB.py' which could be called and ran on a cluster to speed up running time.  
