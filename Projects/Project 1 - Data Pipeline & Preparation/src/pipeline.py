import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

#Optional
from IPython.display import display_html

base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir, '..', 'Datasets', 'raw', 'automobileEDA_dirty_training.csv')

freq_encoding= ['make', 'fuel-system', 'engine-type', 'body-style']
lab_encoding= ['num-of-cylinders', 'symboling', 'horsepower-binned']
oh_encoding= ['drive-wheels', 'num-of-doors', 'aspiration', 'engine-location', 'diesel']
scaler = MinMaxScaler()

#Read Data
def read_data(data):
    df = pd.read_csv(data)
    return df

def anomaly_detection(data):
    df = data.copy()
    numerical_features = df.select_dtypes(include=[np.number]).columns
    categorical_features = df.select_dtypes(include=[object]).columns
    print('===========================================================')
    print('=                     DATASET SUMMARY                     =')
    print('===========================================================')
    print("Rows: {}, Columns: {}".format(df.shape[0], df.shape[1]))
    missing_count = df.isnull().sum()
    print("\nMissing values:\n" + str(missing_count[missing_count > 0].sort_values(ascending=False).to_string()))
    print("\nDuplicate records: {}".format(df.duplicated().sum()))
    print("\nNumerical features: {}".format(len(numerical_features)))
    print("Categorical features: {}".format(len(categorical_features)))
    print('\nData that need to be reformed the format (Date & Case)')
    print(df[['transaction_date', 'make']].head(5))
    print('\nFeature that need to be excluded from Datasets: transaction_date & gas (because it is overlap data with diesel)')
    print('\nOne of values in Engine-Type Feature is replaced "l" => "Inline"')
    return df

def cleansing_data(data):
    df_raw = data.copy()
   # numerical_features = df_raw.select_dtypes(include=[np.number]).columns
   # categorical_features = df_raw.select_dtypes(include=[object]).columns
    
    #Handling Uneven string format
    for cat in df_raw.select_dtypes(include=[object]).columns:
        df_raw[cat] = df_raw[cat].astype(str).str.lower().str.strip()
    #Handling Datetime format
    df_raw['transaction_date'] = pd.to_datetime(df_raw['transaction_date'], format='mixed', dayfirst=True)
    #Drop unnecessary Features
    df_raw = df_raw.drop(columns=['gas', 'transaction_date'])
    #Rename Values
    df_raw['engine-type'] = df_raw['engine-type'].replace({'l': 'inline'})
    #Drop Duplicate Values
    df_raw = df_raw.drop_duplicates(keep='first')
    #Replace string type to real null data
    df_raw = df_raw.replace('nan', np.nan)
    #Fill categorical missing values
    for cat in df_raw.select_dtypes(include=[object]).columns:
        df_raw[cat] = df_raw[cat].fillna(df_raw[cat].mode()[0])
    #Fill numerical missing values
    for num in df_raw.select_dtypes(include=[np.number]).columns:
        df_raw[num] = df_raw[num].fillna(df_raw[num].median())
    bef_missing_count = data.isnull().sum()
    af_missing_count = df_raw.isnull().sum()
    print('===========================================================')
    print('=                     DATA CLEANSING                      =')
    print('===========================================================')
    print("=== Data before Cleansing ======== Data after Cleansing === \n\n    Rows: {}, Columns: {}    |   Rows: {}, Columns: {}".format(data.shape[0], data.shape[1], df_raw.shape[0], df_raw.shape[1]))
    print('===========================================================')
    print('=                      MISSING DATA                       =')
    print('===========================================================')
    print('=== Before Cleansing ===\n')
    print(bef_missing_count[bef_missing_count > 0].sort_values(ascending=False))
    print('\n=== After Cleansing ===\n')
    print(af_missing_count[af_missing_count > 0].sort_values(ascending=False))
    print('\n===========================================================')
    print('=                    DUPLICATE DATA                       =')
    print('===========================================================')
    print("====== before Cleansing ============ After Cleansing ====== \n\n    Duplicate Records: {}     |     Duplicate Records: {} ".format(data.duplicated().sum(),df_raw.duplicated().sum()))
    print('\n===========================================================')
    print('=                    UPDATED COLUMN                       =')
    print('===========================================================')
    print('transaction_date -> deleted\ncategorical features -> Lower Case\nEngine-Type -> renamed 1 of the value\ngas -> deleted')
    
    return df_raw

def transformation_data(df_raw):
    numerical_features = df_raw.select_dtypes(include=[np.number]).columns
    categorical_features = df_raw.select_dtypes(include=[object]).columns
    #Transform numerical data
    df_raw[numerical_features] = scaler.fit_transform(df_raw[numerical_features])
    
    #Transform categorical data
    #One Hot Encoding
    df_raw = pd.get_dummies(df_raw, columns=oh_encoding, drop_first=True)
    #Frequency Encoding 
    for col in freq_encoding:
        freq = df_raw[col].value_counts(normalize=True)
        df_raw[col] = df_raw[col].map(freq)
    #Label Encoding
    #Mapping for Label Encoding
    cylinder_map = {
        'two': 2, 'three': 3, 'four': 4, 'five': 5, 
        'six': 6, 'eight': 8, 'twelve': 12
    }
    horsepower_map = {
        'low': 0, 
        'medium': 1, 
        'high': 2
    }
    df_raw['symboling'] = df_raw['symboling'].astype('int')

    df_raw['num-of-cylinders'] = df_raw['num-of-cylinders'].map(cylinder_map)
    df_raw['horsepower-binned'] = df_raw['horsepower-binned'].map(horsepower_map)
    return df_raw

def save_data(data_raw):
    output_path = os.path.join(base_dir, '..', 'Datasets', 'processed', 'automobileEDA_processed.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data_raw.to_csv(output_path, index= False)
    
def pipeline_data(data):
    df = read_data(data)
    df_detection = anomaly_detection(df)
    df_raw = cleansing_data(df_detection)
    df_final = transformation_data(df_raw)
    result = save_data(df_final)
    
pipeline_data(file_path)