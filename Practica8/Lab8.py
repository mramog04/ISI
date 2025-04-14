#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 8 12:52:38 2025

@author: mramog04

This script performs classification on the Pima Indians Diabetes dataset
using multiple classifiers (Naive Bayes, KNN, Logistic Regression, SVM)
and compares their performance.
"""

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import ConfusionMatrixDisplay
import os

from sklearn.tree import DecisionTreeClassifier



# -------------
# FUNCTIONS
# -------------

def train_classifier(clf, X_train, y_train, X_test, y_test, name):
    """
    Train a classifier and evaluate its performance
    
    Parameters
    ----------
    clf : classifier object
        The classifier to train
    X_train : numpy array
        Training features
    y_train : numpy array
        Training labels
    X_test : numpy array
        Test features
    y_test : numpy array
        Test labels
    name : string
        Name of the classifier for printing
        
    Returns
    -------
    accuracy : float
        Classification accuracy
    f_score : float
        F1 score
    cm : numpy array
        Confusion matrix
    """
    # Train the classifier
    clf.fit(X_train, y_train.ravel())
    
    # Make predictions
    y_pred = clf.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f_score = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    # Print results
    print(f"***** {name} Results *****")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f_score:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("\n")
    
    # Plot confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - {name}')
    plt.show()
    
    return accuracy, f_score, cm

def cross_validate(clf, X, y, n_splits=10, name="Model"):
    """
    Perform stratified k-fold cross-validation
    
    Parameters
    ----------
    clf : classifier object
        The classifier to cross-validate
    X : numpy array
        Features
    y : numpy array
        Labels
    n_splits : int
        Number of folds
    name : string
        Name of the classifier for printing
        
    Returns
    -------
    accuracies : numpy array
        Accuracies for each fold
    f_scores : numpy array
        F1 scores for each fold
    """
    # Initialize arrays to store results
    accuracies = np.zeros(n_splits)
    f_scores = np.zeros(n_splits)
    
    # Create stratified k-fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Perform cross-validation
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        # Split data
        X_train_fold = X[train_index]
        X_test_fold = X[test_index]
        y_train_fold = y[train_index]
        y_test_fold = y[test_index]
        
        # Standardize features based on training set
        scaler = StandardScaler()
        X_train_fold = scaler.fit_transform(X_train_fold)
        X_test_fold = scaler.transform(X_test_fold)
        
        # Train the classifier
        clf.fit(X_train_fold, y_train_fold.ravel())
        
        # Make predictions
        y_pred = clf.predict(X_test_fold)
        
        # Calculate metrics
        accuracies[i] = accuracy_score(y_test_fold, y_pred)
        f_scores[i] = f1_score(y_test_fold, y_pred)
    
    # Print results
    print(f"***** {name} Cross-Validation Results (k={n_splits}) *****")
    print(f"Mean Accuracy: {np.mean(accuracies):.4f}, Std: {np.std(accuracies):.4f}")
    print(f"Mean F1 Score: {np.mean(f_scores):.4f}, Std: {np.std(f_scores):.4f}")
    
    # Print fold-by-fold results
    print("\nFold\tAccuracy\tF1 Score")
    for i in range(n_splits):
        print(f"{i+1}\t{accuracies[i]:.4f}\t\t{f_scores[i]:.4f}")
    print("\n")
    
    return accuracies, f_scores

if __name__ == "__main__":
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Load the dataset
    try:
        data = pd.read_csv("Pima_indian_diabetes.csv")
        print(f"Dataset loaded successfully with {data.shape[0]} rows and {data.shape[1]} columns")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        exit(1)
    
    # Display basic information about the dataset
    print("\nDataset Information:")
    print(data.describe())
    print("\nMissing values per column:")
    print(data.isnull().sum())
    
    # Check for zero values in columns where zero is not meaningful
    print("\nZero values in columns where zero is physiologically implausible:")
    columns_to_check = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for column in columns_to_check:
        zeros = (data[column] == 0).sum()
        print(f"{column}: {zeros} zeros ({zeros/len(data)*100:.1f}%)")
    
    # Option to replace zeros with NaN and then impute
    # For now, we'll keep them as is for simplicity
    
    # Prepare data
    X = data.drop('Outcome', axis=1).values
    y = data['Outcome'].values.reshape(-1, 1)
    
    # Split into training and test sets (70% training, 30% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    print(f"\nTraining set size: {X_train.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples")
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # -------------
    # NAIVE BAYES CLASSIFIER
    # -------------
    print("\n" + "="*50)
    print("NAIVE BAYES CLASSIFIER")
    print("="*50)
    
    # Create classifier
    nb_classifier = GaussianNB()
    
    # Regular training and evaluation
    print("\n--- Regular Training ---")
    nb_acc, nb_f1, nb_cm = train_classifier(nb_classifier, X_train_scaled, y_train, X_test_scaled, y_test, "Naive Bayes")
    
    # K-fold cross-validation
    print("\n--- K-fold Cross-Validation ---")
    nb_cv_acc, nb_cv_f1 = cross_validate(GaussianNB(), X, y, n_splits=10, name="Naive Bayes")
    
    # -------------
    # K-NEAREST NEIGHBORS CLASSIFIER
    # -------------
    print("\n" + "="*50)
    print("K-NEAREST NEIGHBORS CLASSIFIER")
    print("="*50)
    
    # Create classifier (k=5)
    knn_classifier = KNeighborsClassifier(n_neighbors=10)
    
    # Regular training and evaluation
    print("\n--- Regular Training ---")
    knn_acc, knn_f1, knn_cm = train_classifier(knn_classifier, X_train_scaled, y_train, X_test_scaled, y_test, "K-Nearest Neighbors (k=5)")
    
    # K-fold cross-validation
    print("\n--- K-fold Cross-Validation ---")
    knn_cv_acc, knn_cv_f1 = cross_validate(KNeighborsClassifier(n_neighbors=5), X, y, n_splits=10, name="K-Nearest Neighbors (k=5)")
    
    # -------------
    # LOGISTIC REGRESSION CLASSIFIER
    # -------------
    print("\n" + "="*50)
    print("LOGISTIC REGRESSION CLASSIFIER")
    print("="*50)
    
    # Create classifier
    lr_classifier = LogisticRegression(max_iter=1000, random_state=42)
    
    # Regular training and evaluation
    print("\n--- Regular Training ---")
    lr_acc, lr_f1, lr_cm = train_classifier(lr_classifier, X_train_scaled, y_train, X_test_scaled, y_test, "Logistic Regression")
    
    # K-fold cross-validation
    print("\n--- K-fold Cross-Validation ---")
    lr_cv_acc, lr_cv_f1 = cross_validate(LogisticRegression(max_iter=1000, random_state=42), X, y, n_splits=10, name="Logistic Regression")
    
    # -------------
    # SUPPORT VECTOR MACHINE CLASSIFIER
    # -------------
    print("\n" + "="*50)
    print("SUPPORT VECTOR MACHINE CLASSIFIER")
    print("="*50)
    
    # Create classifier with linear kernel
    svm_classifier = SVC(kernel='linear', random_state=42)
    
    # Regular training and evaluation
    print("\n--- Regular Training ---")
    svm_acc, svm_f1, svm_cm = train_classifier(svm_classifier, X_train_scaled, y_train, X_test_scaled, y_test, "SVM (Linear Kernel)")
    
    # K-fold cross-validation
    print("\n--- K-fold Cross-Validation ---")
    svm_cv_acc, svm_cv_f1 = cross_validate(SVC(kernel='linear', random_state=42), X, y, n_splits=10, name="SVM (Linear Kernel)")
    
    
    #--------------
    # Multilayer Perceptron Classifier
    #--------------
    
    # Create classifier
    mlp_classifier = MLPClassifier(hidden_layer_sizes=(100,), max_iter=10000, random_state=42)
    
    # Regular training and evaluation
    print("\n" + "="*50)
    print("MULTILAYER PERCEPTRON CLASSIFIER")
    print("="*50)
    print("\n--- Regular Training ---")
    mlp_acc, mlp_f1, mlp_cm = train_classifier(mlp_classifier, X_train_scaled, y_train, X_test_scaled, y_test, "MLP Classifier")
    
    # K-fold cross-validation
    print("\n--- K-fold Cross-Validation ---")
    mlp_cv_acc, mlp_cv_f1 = cross_validate(MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42), X, y, n_splits=10, name="MLP Classifier")
    # Print results
    print("\n" + "="*50)
    
    #--------------
    # Decision Tree Classifier
    #--------------
    # Create classifier
    dt_classifier = DecisionTreeClassifier(random_state=42)
    
    # Regular training and evaluation
    print("\n" + "="*50)
    print("DECISION TREE CLASSIFIER")
    print("="*50)
    print("\n--- Regular Training ---")
    dt_acc, dt_f1, dt_cm = train_classifier(dt_classifier, X_train_scaled, y_train, X_test_scaled, y_test, "Decision Tree Classifier")
    # K-fold cross-validation
    print("\n--- K-fold Cross-Validation ---")
    dt_cv_acc, dt_cv_f1 = cross_validate(DecisionTreeClassifier(random_state=42), X, y, n_splits=10, name="Decision Tree Classifier")
    # Print results
    print("\n" + "="*50)
    
    #--------------
    # Random Forest Classifier
    #--------------
    # Create classifier
    rf_classifier = RandomForestClassifier(random_state=42)
    # Regular training and evaluation
    print("\n" + "="*50)
    print("RANDOM FOREST CLASSIFIER")
    print("="*50)
    print("\n--- Regular Training ---")
    rf_acc, rf_f1, rf_cm = train_classifier(rf_classifier, X_train_scaled, y_train, X_test_scaled, y_test, "Random Forest Classifier")
    # K-fold cross-validation
    print("\n--- K-fold Cross-Validation ---")
    rf_cv_acc, rf_cv_f1 = cross_validate(RandomForestClassifier(random_state=42), X, y, n_splits=10, name="Random Forest Classifier")
    
    # -------------
    # ORDINARY LEAST SQUARES REGRESSION
    # -------------
    # Create classifier
    ols_classifier = LogisticRegression(max_iter=1000, random_state=42)
    # Regular training and evaluation
    print("\n" + "="*50)
    print("ORDINARY LEAST SQUARES REGRESSION")
    print("="*50)
    print("\n--- Regular Training ---")
    ols_acc, ols_f1, ols_cm = train_classifier(ols_classifier, X_train_scaled, y_train, X_test_scaled, y_test, "Ordinary Least Squares Regression")
    # K-fold cross-validation
    print("\n--- K-fold Cross-Validation ---")
    ols_cv_acc, ols_cv_f1 = cross_validate(LogisticRegression(max_iter=1000, random_state=42), X, y, n_splits=10, name="Ordinary Least Squares Regression")
    
    # -------------
    # Passive Aggressive Classifier
    # -------------
    # Create classifier
    pac_classifier = PassiveAggressiveClassifier(max_iter=1000, random_state=42)
    # Regular training and evaluation
    print("\n" + "="*50)
    print("PASSIVE AGGRESSIVE CLASSIFIER")
    print("="*50)
    print("\n--- Regular Training ---")
    pac_acc, pac_f1, pac_cm = train_classifier(pac_classifier, X_train_scaled, y_train, X_test_scaled, y_test, "Passive Aggressive Classifier")
    # K-fold cross-validation
    print("\n--- K-fold Cross-Validation ---")
    pac_cv_acc, pac_cv_f1 = cross_validate(PassiveAggressiveClassifier(max_iter=1000, random_state=42), X, y, n_splits=10, name="Passive Aggressive Classifier")
    
    # -------------
    # Gaussian Process Classifier
    # -------------
    # Create classifier
    gp_classifier = GaussianProcessClassifier(random_state=42)
    # Regular training and evaluation
    print("\n" + "="*50)
    print("GAUSSIAN PROCESS CLASSIFIER")
    print("="*50)
    print("\n--- Regular Training ---")
    gp_acc, gp_f1, gp_cm = train_classifier(gp_classifier, X_train_scaled, y_train, X_test_scaled, y_test, "Gaussian Process Classifier")
    # K-fold cross-validation
    print("\n--- K-fold Cross-Validation ---")
    gp_cv_acc, gp_cv_f1 = cross_validate(GaussianProcessClassifier(random_state=42), X, y, n_splits=10, name="Gaussian Process Classifier")
    
    # -------------
    # Bagging Classifier
    # -------------
    # Create classifier
    bg_classifier = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=10, random_state=42)
    # Regular training and evaluation
    print("\n" + "="*50)
    print("BAGGING CLASSIFIER")
    print("="*50)
    print("\n--- Regular Training ---")
    bg_acc, bg_f1, bg_cm = train_classifier(bg_classifier, X_train_scaled, y_train, X_test_scaled, y_test, "Bagging Classifier")
    # K-fold cross-validation
    print("\n--- K-fold Cross-Validation ---")
    bg_cv_acc, bg_cv_f1 = cross_validate(BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=10, random_state=42), X, y, n_splits=10, name="Bagging Classifier")
    
    # -------------
    # Gradient Boosting Classifier
    # -------------
    # Create classifier
    gradient_boost_classifier = GradientBoostingClassifier(random_state=42)
    # Regular training and evaluation
    print("\n" + "="*50)
    print("GRADIENT BOOST CLASSIFIER")
    print("="*50)
    print("\n--- Regular Training ---")
    isotonic_acc, isotonic_f1, isotonic_cm = train_classifier(gradient_boost_classifier, X_train_scaled, y_train, X_test_scaled, y_test, "Gradient Boosting Classifier")
    # K-fold cross-validation
    print("\n--- K-fold Cross-Validation ---")
    isotonic_cv_acc, isotonic_cv_f1 = cross_validate(GradientBoostingClassifier(random_state=42), X, y, n_splits=10, name="Gradient Boosting Classifier")
    
    
    
    
    # -------------
    # COMPARISON OF ALL MODELS
    # -------------
    print("\n" + "="*50)
    print("COMPARISON OF ALL CLASSIFIERS")
    print("="*50)
    
    # Create a dataframe to compare results
    results_df = pd.DataFrame({
        'Classifier': ['Naive Bayes', 'K-Nearest Neighbors', 'Logistic Regression', 'SVM','MLP CLF', 'Decision Tree', 'Random Forest', 'Ordinary Least Squares', 'Passive Aggressive', 'Gaussian Process','Bagging Classifier','Isotonic Classifier'],
        'Test Accuracy': [nb_acc, knn_acc, lr_acc, svm_acc,mlp_acc, dt_acc, rf_acc, ols_acc,pac_acc, gp_acc, bg_acc,isotonic_acc],
        'Test F1 Score': [nb_f1, knn_f1, lr_f1, svm_f1,mlp_f1, dt_f1, rf_f1, ols_f1,pac_f1, gp_f1, bg_f1,isotonic_f1],
        'CV Mean Accuracy': [np.mean(nb_cv_acc), np.mean(knn_cv_acc), np.mean(lr_cv_acc), np.mean(svm_cv_acc), np.mean(mlp_cv_acc), np.mean(dt_cv_acc), np.mean(rf_cv_acc), np.mean(ols_cv_acc), np.mean(pac_cv_acc), np.mean(gp_cv_acc), np.mean(bg_cv_acc), np.mean(isotonic_cv_acc)],
        'CV Mean F1 Score': [np.mean(nb_cv_f1), np.mean(knn_cv_f1), np.mean(lr_cv_f1), np.mean(svm_cv_f1), np.mean(mlp_cv_f1), np.mean(dt_cv_f1), np.mean(rf_cv_f1), np.mean(ols_cv_f1), np.mean(pac_cv_f1), np.mean(gp_cv_f1), np.mean(bg_cv_f1), np.mean(isotonic_cv_f1)], 
    })
    
    print(results_df.round(4))
    
    # Plot comparison
    plt.figure(figsize=(12, 6))
    
    # Test metrics
    plt.subplot(1, 2, 1)
    x = np.arange(len(results_df['Classifier']))
    width = 0.35
    plt.bar(x - width/2, results_df['Test Accuracy'], width, label='Accuracy')
    plt.bar(x + width/2, results_df['Test F1 Score'], width, label='F1 Score')
    plt.xlabel('Classifier')
    plt.ylabel('Score')
    plt.title('Test Performance')
    plt.xticks(x, results_df['Classifier'], rotation=45)
    plt.ylim(0, 1)
    plt.legend()
    
    # CV metrics
    plt.subplot(1, 2, 2)
    plt.bar(x - width/2, results_df['CV Mean Accuracy'], width, label='CV Accuracy')
    plt.bar(x + width/2, results_df['CV Mean F1 Score'], width, label='CV F1 Score')
    plt.xlabel('Classifier')
    plt.ylabel('Score')
    plt.title('Cross-Validation Performance')
    plt.xticks(x, results_df['Classifier'], rotation=45)
    plt.ylim(0, 1)
    plt.legend()
    
    plt.tight_layout()
    plt.show()