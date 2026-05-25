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
    """Kelas model Decision Tree untuk prediksi kanker paru."""

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
    # PREPROCESSING DATA
    # ========================================
    def preprocess(self, filepath):
        """Load dan preprocessing dataset dari CSV."""
        df = pd.read_csv(filepath)

        # 🔧 WAJIB: Bersihkan nama kolom (hapus spasi, ganti jadi underscore)
        df.columns = df.columns.str.strip()
        df.columns = df.columns.str.replace(' ', '_')

        # Encode GENDER: M=1, F=0
        df['GENDER'] = df['GENDER'].map({'M': 1, 'F': 0})

        # Encode LUNG_CANCER: YES=1, NO=0
        df['LUNG_CANCER'] = df['LUNG_CANCER'].map({'YES': 1, 'NO': 0})

        # Simpan data yang sudah diproses
        self.df_processed = df
        self.feature_names = df.drop('LUNG_CANCER', axis=1).columns.tolist()

        return df

    # ========================================
    # TRAINING MODEL
    # ========================================
    def train(self, filepath):
        """Training model Decision Tree."""
        df = self.preprocess(filepath)

        # Pisahkan fitur dan target
        X = df.drop('LUNG_CANCER', axis=1)
        y = df['LUNG_CANCER']

        # Split data: 80% training, 20% testing
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Buat dan training model Decision Tree (SUDAH DITUNING UNTUK KAGGLE)
        self.model = DecisionTreeClassifier(
            max_depth=4,
            min_samples_split=15,
            min_samples_leaf=8,
            random_state=42,
            criterion='gini',
            class_weight='balanced'
        )

        # Split data train dan test
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        # Training model
        self.model.fit(X_train, y_train)

        # Prediksi data testing
        y_pred = self.model.predict(X_test)

        # Simpan confusion matrix
        self.cm = confusion_matrix(y_test, y_pred)

        # Simpan accuracy
        self.accuracy = accuracy_score(y_test, y_pred)

        # Evaluasi
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = self.model.predict(X_test)
        self.accuracy = accuracy_score(y_test, self.y_pred)
        self.report = classification_report(
            y_test, self.y_pred, 
            target_names=['Tidak Kanker', 'Kanker Paru']
        )
        self.cm = confusion_matrix(y_test, self.y_pred)

        print(f"\n✅ Model berhasil di-training!")
        print(f"   Akurasi: {self.accuracy:.2%}")
        print(f"   Confusion Matrix:\n{self.cm}")

        return self.model

    # ========================================
    # PREDIKSI
    # ========================================
    def predict(self, input_dict):
        """
        Prediksi berdasarkan input dictionary.
        """
        if self.model is None:
            raise Exception("Model belum di-training! Jalankan train() dulu.")

        # Buat DataFrame dari input
        input_df = pd.DataFrame([input_dict])

        # 🔧 WAJIB: Bersihkan nama kolom dari input user juga!
        input_df.columns = input_df.columns.str.strip()
        input_df.columns = input_df.columns.str.replace(' ', '_')

        # Pastikan urutan kolom sesuai dengan saat training
        input_df = input_df[self.feature_names]

        # Prediksi
        prediction = self.model.predict(input_df)[0]
        probability = self.model.predict_proba(input_df)[0]

        return prediction, probability

    # ========================================
    # VISUALISASI DECISION TREE
    # ========================================
    def get_tree_plot(self):
        """Mengembalikan figure matplotlib visualisasi decision tree."""
        fig, ax = plt.subplots(figsize=(22, 12))
        
        plot_tree(
            self.model,
            feature_names=self.feature_names,
            class_names=['Tidak Kanker', 'Kanker Paru'],
            filled=True,
            rounded=True,
            fontsize=9,
            proportion=True,
            impurity=True
        )
        
        plt.title(
            "Visualisasi Decision Tree - Prediksi Kanker Paru",
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
        """Mengembalikan DataFrame feature importance."""
        importance = self.model.feature_importances_
        
        label_map = {
            'GENDER': 'Jenis Kelamin',
            'AGE': 'Usia',
            'SMOKING': 'Merokok',
            'YELLOW_FINGERS': 'Jari Kuning',
            'ANXIETY': 'Kecemasan',
            'PEER_PRESSURE': 'Tekanan Teman Sebaya',
            'CHRONIC_DISEASE': 'Penyakit Kronis',
            'FATIGUE': 'Kelelahan',
            'ALLERGY': 'Alergi',
            'WHEEZING': 'Mengi/Napas Berbunyi',
            'ALCOHOL_CONSUMING': 'Konsumsi Alkohol',
            'COUGHING': 'Batuk',
            'SHORTNESS_OF_BREATH': 'Sesak Napas',
            'SWALLOWING_DIFFICULTY': 'Sulit Menelan',
            'CHEST_PAIN': 'Nyeri Dada'
        }

        fi_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Label': [label_map.get(f, f) for f in self.feature_names],
            'Importance': importance
        }).sort_values('Importance', ascending=True)

        # Filter hanya yang importance > 0
        fi_df = fi_df[fi_df['Importance'] > 0]

        return fi_df

    # ========================================
    # CONFUSION MATRIX
    # ========================================
    def get_confusion_matrix_plot(self):
        """Mengembalikan figure matplotlib confusion matrix."""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.heatmap(
            self.cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Tidak Kanker', 'Kanker Paru'],
            yticklabels=['Tidak Kanker', 'Kanker Paru'],
            ax=ax,
            linewidths=1,
            linecolor='white'
        )
        
        ax.set_xlabel('Prediksi', fontsize=12, fontweight='bold')
        ax.set_ylabel('Aktual', fontsize=12, fontweight='bold')
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold', pad=15)
        
        plt.tight_layout()
        return fig

    # ========================================
    # INFO MODEL
    # ========================================
    def get_model_info(self):
        """Mengembalikan informasi detail model."""
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