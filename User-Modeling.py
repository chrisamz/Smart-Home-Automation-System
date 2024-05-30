# user_modeling.py

"""
User Modeling Module for Smart Home Automation System

This module contains functions for developing user models to understand and predict user preferences
and behaviors in a smart home environment.

Techniques Used:
- Clustering
- Classification
- Behavioral Analysis

Algorithms Used:
- K-Means
- Decision Trees

Libraries/Tools:
- scikit-learn
"""

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

class UserModeling:
    def __init__(self, n_clusters=3):
        """
        Initialize the UserModeling class.
        
        :param n_clusters: int, number of clusters for K-Means
        """
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        self.decision_tree = DecisionTreeClassifier(random_state=42)

    def load_data(self, filepath):
        """
        Load data from a CSV file.
        
        :param filepath: str, path to the CSV file
        :return: DataFrame, loaded data
        """
        return pd.read_csv(filepath, parse_dates=['timestamp'])

    def preprocess_data(self, data):
        """
        Preprocess the data for clustering and classification.
        
        :param data: DataFrame, input data
        :return: DataFrame, preprocessed data
        """
        # Fill missing values
        data = data.fillna(data.mean())
        return data

    def cluster_users(self, data):
        """
        Cluster users into different behavior groups using K-Means.
        
        :param data: DataFrame, input data
        :return: DataFrame, data with cluster labels
        """
        features = data.drop(columns=['user_id'])
        data['user_cluster'] = self.kmeans.fit_predict(features)
        return data

    def train_classifier(self, data):
        """
        Train a Decision Tree classifier to predict user behavior clusters.
        
        :param data: DataFrame, input data with cluster labels
        """
        X = data.drop(columns=['user_id', 'user_cluster'])
        y = data['user_cluster']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.decision_tree.fit(X_train, y_train)
        y_pred = self.decision_tree.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        print("Decision Tree Classifier Accuracy:", accuracy)
        print("Classification Report:\n", report)

    def save_models(self):
        """
        Save the trained models to files.
        """
        joblib.dump(self.kmeans, 'models/kmeans_model.pkl')
        joblib.dump(self.decision_tree, 'models/decision_tree_model.pkl')
        print("Models saved to 'models/kmeans_model.pkl' and 'models/decision_tree_model.pkl'.")

    def load_models(self):
        """
        Load the trained models from files.
        """
        self.kmeans = joblib.load('models/kmeans_model.pkl')
        self.decision_tree = joblib.load('models/decision_tree_model.pkl')

    def classify_new_user(self, user_data):
        """
        Classify a new user into a behavior group using the trained Decision Tree classifier.
        
        :param user_data: DataFrame, data of the new user
        :return: int, predicted behavior group
        """
        return self.decision_tree.predict(user_data)[0]

if __name__ == "__main__":
    data_filepath = 'data/processed/preprocessed_smart_home_data.csv'

    user_modeler = UserModeling()
    data = user_modeler.load_data(data_filepath)
    data = user_modeler.preprocess_data(data)

    # Cluster users and train classifier
    clustered_data = user_modeler.cluster_users(data)
    user_modeler.train_classifier(clustered_data)

    # Save models
    user_modeler.save_models()
