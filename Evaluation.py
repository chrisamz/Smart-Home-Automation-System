# evaluation.py

"""
Evaluation Module for Smart Home Automation System

This module contains functions for evaluating the performance of user modeling,
reinforcement learning, and automation algorithms using appropriate metrics.

Techniques Used:
- Model Evaluation
- Automation System Evaluation

Metrics Used:
- Accuracy
- Precision
- Recall
- F1-score
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

Libraries/Tools:
- scikit-learn
- tensorflow
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error
import joblib
import tensorflow as tf

class ModelEvaluation:
    def __init__(self):
        """
        Initialize the ModelEvaluation class.
        """
        pass

    def load_data(self, filepath):
        """
        Load test data from a CSV file.
        
        :param filepath: str, path to the CSV file
        :return: DataFrame, loaded data
        """
        return pd.read_csv(filepath)

    def load_model(self, model_filepath):
        """
        Load a trained model from a file.
        
        :param model_filepath: str, path to the saved model
        :return: model, loaded model
        """
        return joblib.load(model_filepath)

    def evaluate_classification(self, model, X_test, y_test):
        """
        Evaluate a classification model using accuracy, precision, recall, and F1-score.
        
        :param model: trained model
        :param X_test: DataFrame, testing features
        :param y_test: Series, testing target
        :return: dict, evaluation metrics
        """
        y_pred = model.predict(X_test)
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted')
        }
        return metrics

    def evaluate_regression(self, model, X_test, y_test):
        """
        Evaluate a regression model using MAE and RMSE.
        
        :param model: trained model
        :param X_test: DataFrame, testing features
        :param y_test: Series, testing target
        :return: dict, evaluation metrics
        """
        y_pred = model.predict(X_test)
        metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
        }
        return metrics

    def evaluate_reinforcement_learning(self, model, scaler, X_test, y_test):
        """
        Evaluate a reinforcement learning system using MAE and RMSE.
        
        :param model: trained model
        :param scaler: scaler used for preprocessing
        :param X_test: DataFrame, testing features
        :param y_test: Series, testing target
        :return: dict, evaluation metrics
        """
        X_test = scaler.transform(X_test)
        y_pred = model.predict(X_test)
        metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
        }
        return metrics

if __name__ == "__main__":
    test_data_filepath = 'data/processed/preprocessed_test_data.csv'
    user_model_target_column = 'user_cluster'
    rl_model_target_column = 'desired_action'

    evaluator = ModelEvaluation()
    data = evaluator.load_data(test_data_filepath)

    # Evaluate User Modeling
    user_model = evaluator.load_model('models/decision_tree_model.pkl')
    X_test_user = data.drop(columns=[user_model_target_column])
    y_test_user = data[user_model_target_column]
    user_model_metrics = evaluator.evaluate_classification(user_model, X_test_user, y_test_user)
    print("User Modeling Evaluation:", user_model_metrics)

    # Evaluate Reinforcement Learning
    scaler = joblib.load('models/scaler.pkl')
    rl_model = tf.keras.models.load_model('models/reinforcement_learning_model.h5')
    X_test_rl = data.drop(columns=[rl_model_target_column])
    y_test_rl = data[rl_model_target_column]
    rl_model_metrics = evaluator.evaluate_reinforcement_learning(rl_model, scaler, X_test_rl, y_test_rl)
    print("Reinforcement Learning Evaluation:", rl_model_metrics)
