# automation_algorithm.py

"""
Automation Algorithm Module for Smart Home Automation System

This module contains functions for developing an automation algorithm to manage various aspects
of the smart home, such as lighting and climate control, based on learned user preferences and real-time data.

Techniques Used:
- Rule-based systems
- Machine learning models
- Reinforcement learning

Libraries/Tools:
- scikit-learn
- TensorFlow
- Keras

"""

import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from sklearn.tree import DecisionTreeClassifier

class AutomationAlgorithm:
    def __init__(self):
        """
        Initialize the AutomationAlgorithm class.
        """
        self.user_model = None
        self.rl_model = None
        self.scaler = None

    def load_models(self):
        """
        Load the trained models from files.
        """
        self.user_model = joblib.load('models/decision_tree_model.pkl')
        self.rl_model = tf.keras.models.load_model('models/reinforcement_learning_model.h5')
        self.scaler = joblib.load('models/scaler.pkl')
        print("Models loaded successfully.")

    def preprocess_data(self, data):
        """
        Preprocess the data for prediction.
        
        :param data: DataFrame, input data
        :return: DataFrame, preprocessed data
        """
        data = data.fillna(data.mean())
        data = self.scaler.transform(data)
        return data

    def predict_user_preferences(self, user_data):
        """
        Predict user preferences using the trained user model.
        
        :param user_data: DataFrame, data of the user
        :return: int, predicted user preference
        """
        return self.user_model.predict(user_data)[0]

    def predict_next_action(self, state):
        """
        Predict the next best action for the current state using the RL model.
        
        :param state: DataFrame, current state data
        :return: int, predicted action
        """
        state = self.scaler.transform(state)
        action_values = self.rl_model.predict(state)
        return np.argmax(action_values[0])

    def automate(self, user_data, state_data):
        """
        Automate the smart home based on user preferences and current state.
        
        :param user_data: DataFrame, data of the user
        :param state_data: DataFrame, current state data
        :return: dict, actions to perform
        """
        user_preference = self.predict_user_preferences(user_data)
        next_action = self.predict_next_action(state_data)
        
        actions = {
            'lighting': 'on' if next_action == 0 else 'off',
            'temperature': 'increase' if next_action == 1 else 'decrease'
        }
        
        print(f"User Preference: {user_preference}, Next Action: {next_action}")
        return actions

if __name__ == "__main__":
    user_data_filepath = 'data/processed/user_data.csv'
    state_data_filepath = 'data/processed/state_data.csv'

    automation = AutomationAlgorithm()
    automation.load_models()

    user_data = pd.read_csv(user_data_filepath)
    state_data = pd.read_csv(state_data_filepath)

    user_data = automation.preprocess_data(user_data)
    state_data = automation.preprocess_data(state_data)

    actions = automation.automate(user_data, state_data)
    print("Actions to perform:", actions)
