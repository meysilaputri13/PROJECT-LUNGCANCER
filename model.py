"""
model.py
Kelas untuk training, evaluasi, dan prediksi model Decision Tree.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix
)
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import os


class LungCancerModel:
    """Decision Tree model class for lung cancer prediction."""

    def __init__(self):
        self.model = None
        self.accuracy = 0
        self.report = ""
        self.X_test = None
        self.y_test = None
        self.y_pred = None
        self.feature_names = []
        self.df_processed = None
        self.cm = None

    # ========================================
    # DATA PREPROCESSING
    # ========================================
    def preprocess(self, filepath):
        """Load and preprocess dataset from CSV."""
        df = pd.read_csv(filepath)

        # Clean column names (strip spaces, replace spaces with underscores)
        df.columns = df.columns.str.strip()
        df.columns = df.columns.str.replace(' ', '_')

        # Encode GENDER: M=1, F=0
        df['GENDER'] = df['GENDER'].map({'M': 1, 'F': 0})

        # Encode LUNG_CANCER: YES=1, NO=0
        df['LUNG_CANCER'] = df['LUNG_CANCER'].map({'YES': 1, 'NO': 0})

        # Store processed data
        self.df_processed = df
        self.feature_names = df.drop('LUNG_CANCER', axis=1).columns.tolist()

        return df

    # ========================================
    # MODEL TRAINING
    # ========================================
    def train(self, filepath):
        """Train the Decision Tree model."""
        df = self.preprocess(filepath)

        # Separate features and target
        X = df.drop('LUNG_CANCER', axis=1)
        y = df['LUNG_CANCER']

        # Split data: 80% training, 20% testing
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Create and train Decision Tree model (TUNED FOR KAGGLE DATASET)
        self.model = DecisionTreeClassifier(
            max_depth=4,
            min_samples_split=15,
            min_samples_leaf=8,
            random_state=42,
            criterion='gini',
            class_weight='balanced'
        )

        # Train model
        self.model.fit(X_train, y_train)

        # Predict testing data
        y_pred = self.model.predict(X_test)

        # Evaluation metrics
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = y_pred
        self.accuracy = accuracy_score(y_test, y_pred)
        self.report = classification_report(
            y_test, self.y_pred, 
            target_names=['No Cancer', 'Lung Cancer']
        )
        self.cm = confusion_matrix(y_test, y_pred)

        print(f"\n✅ Model successfully trained!")
        print(f"   Accuracy: {self.accuracy:.2%}")
        print(f"   Confusion Matrix:\n{self.cm}")

        return self.model

    # ========================================
    # PREDICTION
    # ========================================
    def predict(self, input_dict):
        """
        Predict based on input dictionary.
        """
        if self.model is None:
            raise Exception("Model is not trained yet! Run train() first.")

        # Create DataFrame from input
        input_df = pd.DataFrame([input_dict])

        # Clean column names from user input as well
        input_df.columns = input_df.columns.str.strip()
        input_df.columns = input_df.columns.str.replace(' ', '_')

        # Ensure column order matches training data
        input_df = input_df[self.feature_names]

        # Predict
        prediction = self.model.predict(input_df)[0]
        probability = self.model.predict_proba(input_df)[0]

        return prediction, probability

    # ========================================
    # DECISION TREE VISUALIZATION
    # ========================================
    def get_tree_plot(self):
        """Return matplotlib figure of the decision tree visualization."""
        fig, ax = plt.subplots(figsize=(22, 12))
        
        plot_tree(
            self.model,
            feature_names=self.feature_names,
            class_names=['No Cancer', 'Lung Cancer'],
            filled=True,
            rounded=True,
            fontsize=9,
            proportion=True,
            impurity=True
        )
        
        plt.title(
            "Decision Tree Visualization - Lung Cancer Prediction",
            fontsize=16, 
            fontweight='bold',
            color='#1e293b',
            pad=20
        )
        
        plt.tight_layout()
        return fig

    # ========================================
    # FEATURE IMPORTANCE
    # ========================================
    def get_feature_importance(self):
        """Return DataFrame of feature importances."""
        importance = self.model.feature_importances_
        
        label_map = {
            'GENDER': 'Gender',
            'AGE': 'Age',
            'SMOKING': 'Smoking',
            'YELLOW_FINGERS': 'Yellow Fingers',
            'ANXIETY': 'Anxiety',
            'PEER_PRESSURE': 'Peer Pressure',
            'CHRONIC_DISEASE': 'Chronic Disease',
            'FATIGUE': 'Fatigue',
            'ALLERGY': 'Allergy',
            'WHEEZING': 'Wheezing',
            'ALCOHOL_CONSUMING': 'Alcohol Consuming',
            'COUGHING': 'Coughing',
            'SHORTNESS_OF_BREATH': 'Shortness of Breath',
            'SWALLOWING_DIFFICULTY': 'Swallowing Difficulty',
            'CHEST_PAIN': 'Chest Pain'
        }

        fi_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Label': [label_map.get(f, f) for f in self.feature_names],
            'Importance': importance
        }).sort_values('Importance', ascending=True)

        # Filter only importance > 0
        fi_df = fi_df[fi_df['Importance'] > 0]

        return fi_df

    # ========================================
    # CONFUSION MATRIX
    # ========================================
    def get_confusion_matrix_plot(self):
        """Return matplotlib figure of the confusion matrix."""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.heatmap(
            self.cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['No Cancer', 'Lung Cancer'],
            yticklabels=['No Cancer', 'Lung Cancer'],
            ax=ax,
            linewidths=1,
            linecolor='white'
        )
        
        ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
        ax.set_ylabel('Actual', fontsize=12, fontweight='bold')
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold', pad=15)
        
        plt.tight_layout()
        return fig

    # ========================================
    # MODEL INFO
    # ========================================
    def get_model_info(self):
        """Return detailed model information."""
        return {
            'max_depth': self.model.max_depth,
            'min_samples_split': self.model.min_samples_split,
            'min_samples_leaf': self.model.min_samples_leaf,
            'criterion': self.model.criterion,
            'n_features': len(self.feature_names),
            'accuracy': self.accuracy,
            'n_nodes': self.model.tree_.node_count,
            'n_leaves': self.model.tree_.n_leaves,
        }