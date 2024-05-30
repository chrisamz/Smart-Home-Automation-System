# data_preprocessing.py

"""
Data Preprocessing Module for Smart Home Automation System

This module contains functions for collecting, cleaning, normalizing, and preparing
smart home data for further analysis and modeling.

Techniques Used:
- Data cleaning
- Normalization
- Feature extraction
- Handling missing data
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

class DataPreprocessing:
    def __init__(self):
        """
        Initialize the DataPreprocessing class.
        """
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')

    def load_data(self, filepath):
        """
        Load data from a CSV file.
        
        :param filepath: str, path to the CSV file
        :return: DataFrame, loaded data
        """
        return pd.read_csv(filepath, parse_dates=['timestamp'])

    def clean_data(self, data):
        """
        Clean the data by removing duplicates and handling missing values.
        
        :param data: DataFrame, input data
        :return: DataFrame, cleaned data
        """
        data = data.drop_duplicates()
        data = pd.DataFrame(self.imputer.fit_transform(data), columns=data.columns)
        return data

    def normalize_data(self, data, columns):
        """
        Normalize the specified columns in the data.
        
        :param data: DataFrame, input data
        :param columns: list, columns to be normalized
        :return: DataFrame, normalized data
        """
        data[columns] = self.scaler.fit_transform(data[columns])
        return data

    def extract_features(self, data):
        """
        Extract relevant features from the data.
        
        :param data: DataFrame, input data
        :return: DataFrame, data with extracted features
        """
        # Example feature extraction: Calculate average temperature and light intensity
        data['avg_temperature'] = data[['temperature_sensor_1', 'temperature_sensor_2']].mean(axis=1)
        data['avg_light'] = data[['light_sensor_1', 'light_sensor_2']].mean(axis=1)
        return data

    def preprocess(self, filepath, columns_to_normalize):
        """
        Execute the full preprocessing pipeline.
        
        :param filepath: str, path to the input data file
        :param columns_to_normalize: list, columns to be normalized
        :return: DataFrame, preprocessed data
        """
        data = self.load_data(filepath)
        data = self.clean_data(data)
        data = self.extract_features(data)
        data = self.normalize_data(data, columns_to_normalize)
        return data

if __name__ == "__main__":
    smart_home_data_filepath = 'data/raw/smart_home_data.csv'
    columns_to_normalize = ['temperature_sensor_1', 'temperature_sensor_2', 'light_sensor_1', 'light_sensor_2', 'avg_temperature', 'avg_light']

    preprocessing = DataPreprocessing()

    # Preprocess smart home data
    preprocessed_smart_home_data = preprocessing.preprocess(smart_home_data_filepath, columns_to_normalize)
    preprocessed_smart_home_data.to_csv('data/processed/preprocessed_smart_home_data.csv', index=False)
    print("Smart home data preprocessing completed and saved to 'data/processed/preprocessed_smart_home_data.csv'.")
