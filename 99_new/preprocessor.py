import argparse
import os
import warnings

print('Unzipping Athena results')
import subprocess
sub_output = subprocess.getoutput(["""for FILE in /opt/ml/processing/features/*; do gunzip $FILE; done"""])

import pandas as pd
import numpy as np
import glob

from sklearn.model_selection import train_test_split

if __name__=='__main__':
    
    # Read the arguments passed to the script.
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-test-split-ratio', type=float, default=0.3)
    args, _ = parser.parse_known_args()
    
    filenames = glob.glob('/opt/ml/processing/features/*')
    print(filenames)

    dfs = []
    for filename in filenames:
        dfs.append(pd.read_csv(filename, sep=chr(1), header=None))
    df = pd.concat(dfs, ignore_index=True)
    
    df.columns = ["breakdown","wind_speed","rpm_blade","oil_temperature","oil_level","temperature","humidity","vibrations_frequency",
                  "pressure","turbine_id_tid004","turbine_id_tid001","turbine_id_tid006","turbine_id_tid008",
                  "turbine_id_tid002","turbine_id_tid003","turbine_id_tid005","turbine_id_tid009","turbine_id_tid010",
                  "turbine_id_tid007","turbine_type_hawt","turbine_type_vawt","wind_direction_s","wind_direction_n",
                  "wind_direction_w","wind_direction_sw","wind_direction_e","wind_direction_se","wind_direction_ne","wind_direction_nw"]

    X = df.drop('breakdown', axis=1)
    y = df['breakdown']
    
    split_ratio = args.train_test_split_ratio
    print('Splitting data into train and validation sets with ratio {}'.format(split_ratio))
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=split_ratio, random_state=0)

    print('Train features shape after preprocessing: {}'.format(X_train.shape))
    print('Validation features shape after preprocessing: {}'.format(X_val.shape))
    
    # Saving outputs.
    train_features_output_path = os.path.join('/opt/ml/processing/train', 'train_features.csv')
    train_labels_output_path = os.path.join('/opt/ml/processing/train', 'train_labels.csv')
    
    val_features_output_path = os.path.join('/opt/ml/processing/val', 'val_features.csv')
    val_labels_output_path = os.path.join('/opt/ml/processing/val', 'val_labels.csv')
    
    print('Saving training features to {}'.format(train_features_output_path))
    pd.DataFrame(X_train).to_csv(train_features_output_path, header=False, index=False)
    
    print('Saving validation features to {}'.format(val_features_output_path))
    pd.DataFrame(X_val).to_csv(val_features_output_path, header=False, index=False)
    
    print('Saving training labels to {}'.format(train_labels_output_path))
    pd.DataFrame(y_train).to_csv(train_labels_output_path, header=False, index=False)
    
    print('Saving validation labels to {}'.format(val_labels_output_path))
    pd.DataFrame(y_val).to_csv(val_labels_output_path, header=False, index=False)