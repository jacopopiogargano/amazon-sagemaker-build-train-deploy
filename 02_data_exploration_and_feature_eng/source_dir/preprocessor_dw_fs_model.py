import argparse
import os
import warnings

import subprocess
subprocess.call(['pip', 'install', 'sagemaker-experiments'])

import pandas as pd
import numpy as np
import tarfile

from smexperiments.tracker import Tracker

from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import make_column_transformer

from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

columns = ['turbine_id', 'turbine_type', 'wind_speed', 'rpm_blade', 'oil_temperature',
           'oil_level', 'temperature', 'humidity', 'vibrations_frequency', 'pressure', 'wind_direction', 'breakdown']

if __name__=='__main__':
    
    # Creating a tracker to log information during the job execution
    tracker = Tracker.load()
    
    # Read input data into a Pandas dataframe.
    input_data_path = os.path.join('/opt/ml/processing/input', 'windturbine_raw_data_header.csv')
    print('Reading input data from {}'.format(input_data_path))
    df = pd.read_csv(input_data_path)
    df.columns = columns
    
    # Replacing certain null values.
    df['turbine_type'] = df['turbine_type'].fillna("HAWT")
    tracker.log_parameter('default-turbine-type', 'HAWT')
    
    df['oil_temperature'] = df['oil_temperature'].fillna(37.0)
    tracker.log_parameter('default-oil-temperature', 37.0)
    
    # Defining one-hot encoders.
    transformer = make_column_transformer(
        (['turbine_id', 'turbine_type', 'wind_direction'], OneHotEncoder(sparse=False)), remainder="passthrough"
    )
    
    X = df.drop('breakdown', axis=1)
    
    featurizer_model = transformer.fit(X)
    
    # Saving model.
    model_path = os.path.join('/opt/ml/processing/model', 'model.joblib')
    model_output_path = os.path.join('/opt/ml/processing/model', 'model.tar.gz')
    
    print('Saving featurizer model to {}'.format(model_output_path))
    joblib.dump(featurizer_model, model_path)
    tar = tarfile.open(model_output_path, "w:gz")
    tar.add(model_path, arcname="model.joblib")
    tar.close()
    
    tracker.close()
