import flask
from flask_cors import CORS
from flask import jsonify

app = flask.Flask(__name__)
CORS(app)
app.config["DEBUG"] = True

import os
import glob

import pandas as pd
import numpy as np
from subprocess import Popen, PIPE

main_dir = '//SA-MODAT-MTO-PR/Data-Safi/'

def get_gp2_data(nb_rows=30):
    # Find last File
    p = Popen("last_file_gp2.bat", shell=True, stdout=PIPE,)
    stdout, stderr = p.communicate()
    file_path = stdout.decode('utf-8').rstrip()
    
    # Load last File
    data = pd.read_csv(main_dir + file_path,low_memory=False,
                   delimiter='\t',quotechar='"',decimal=',',).dropna()

    # Rename Columns and filter on last 30 min
    data = data.rename(columns={'Unnamed: 0' : 'datetime',
                                'Speed@1m': 'Wind Speed (m.s-1)', 
                                'Dir': 'Wind Direction (deg C)',
                                'AirTemp' : 'Air Temperature (deg C)',
                                "Rad'n" : 'Radiation (W.m-2)',
                                'Rain@1m' : 'Precipitation (mm)'}).tail(nb_rows)
    # Format Day & Time
    data['Day'] = data.datetime.map(lambda x : x[0:10])
    data['Time'] = data.datetime.map(lambda x : x[11:16])
    
    # Filter Columns
    data.drop(columns={'datetime','Pressure','RH','Power'},inplace=True)
    
    return data
    
    

@app.route('/data/gp2_last_row/', methods=['GET'])
def home():
    return jsonify(get_gp2_data(nb_rows=1).to_dict(orient='records'))

@app.route('/data/gp2_last_30min/', methods=['GET'])
def gp2_last_30min():
    return jsonify(get_gp2_data(nb_rows=30).to_dict(orient='records'))

app.run(host='0.0.0.0')