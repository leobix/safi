# Documentation for MIT / SAFI weather forecast project

This documentation serves the purpose to share current running codes and results, which can be turned into a beta version for testing phase. 

Individual functions are coded in separate python files: 
1. ```utils/data_preparation.py``` is the file containing functions to process measurements dataframe (data obtained by Plum air device) as well as forecast dataframe (official forecast from the weather agency). Specifically, it includes functions to: 
    - read raw CSV files for measurement (two per year from 2015 Semester 1 to 2020 Semester 1) and official government forecast data
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
    - t_list: time steps to be predicted, default = [1,3,6,9,12,15,18,21,24,27,30,33,36,39,42,45,48]
    
5. ```main_OCT.py``` is the main function to be called to perform Optimal Trees for regression and classification. You can amend in particular the following arguments:
    - steps-in: number of past data for prediction, default = 48 (hours of data)
    - steps-out: what specific hour do you want to build a model for.
However, you need a license to use it, so the code won't work until then.

# Important facts for deployment in Safi

1. Running these two last files can take a long time to run in a local computer (CPU). They should be called and ran on a cluster to speed up running time.  It takes less than 10 minutes to run ```main_XGB.py``` with 24 CPUs on a cluster. It takes around 12h to run ```main_OCT.py```with 24 CPUs.

2. It is possible to retrain regularly the models and store the trained models for use. A weekly basis or bi-weekly basis for retraining with new data sounds good.

3. When adding new data, it should be in the right format. Please look at the files ```data/forecast_data_15_20.csv``` for official forecast data formatting and ```data/2020S1.csv``` for Plumair device measurements formatting to be copy pasted in these files.

4. For each timestep to predict, 5 models are required: 
    - 1 for wind speed, 1 for cosinus wind direction, 1 for sinus wind direction: these 3 models can then be used to get a scenario with the policy decided by Safi team
    - 1 for direct scenario classification, 1 for direct dangerous scenario classification: directly predicting the scenario yields better accuracy than passing through wind speed and wind direction predictions. The use of these models for confirmation or additional alert seems valuable.
    
5. We propose to use XGBoost which is a state-of-the-art method. We also propose Optimal Regression and Classification Trees that are equally good or better and also present the advantage of being interpretable. However, a license from InterpretableAI is required.
