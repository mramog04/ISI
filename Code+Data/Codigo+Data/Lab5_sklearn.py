#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 16:12:41 2022
Modified on Mon Mar 13 2023
Modified on Thu Mar 13 2025

@author: CHANGE THE NAME

This script carries out a classification experiment of the spambase dataset by
means of the kNN classifier, USING THE SCIKIT-LEARN PACKAGE
"""

# Import whatever else you need to import
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt



# 
# -------------
# MAIN PROGRAM
# -------------
if __name__ == "__main__":


    # Load csv with data into a pandas dataframe
    """
    Each row in this dataframe contains a feature vector, which represents an
    email.
    Each column represents a variable, EXCEPT LAST COLUMN, which represents
    the true class of the corresponding element (i.e. row): 1 means "spam",
    and 0 means "not spam"
    """
    dir_data = "Data"
    spam_df = pd.read_csv("spambase_data.csv")
    y_df = spam_df[['Class']].copy()
    X_df = spam_df.copy()
    X_df = X_df.drop('Class', axis=1)

    # Convert dataframe to numpy array
    X = X_df.to_numpy()
    y = y_df.to_numpy()
    
    """
    Parameter that indicates the proportion of elements that the test set will
    have
    """
    proportion_test = 0.3

    """
    Partition of the dataset into training and test sets is done. 
    Use the function train_test_split from scikit_learn
    """
    # ====================== YOUR CODE HERE ======================
    num_elements_train = int(spam_df.shape[0]* (1-proportion_test))
    num_elements_test = spam_df.shape[0] - num_elements_train
    
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=proportion_test)
    
    # ============================================================

    """
    Create an instance of the kNN classifier using scikit-learn
    """
    # ====================== YOUR CODE HERE ======================
    knn = KNeighborsClassifier(n_neighbors=5)
    # ============================================================

    """
    Train the classifier
    """
    # ====================== YOUR CODE HERE ======================
    knn.fit(X_train,y_train)
    # ============================================================

    """
    Get the predictions for the test set samples given by the classifier
    """
    # ====================== YOUR CODE HERE ======================
    knn.predict(X_test)
    # ============================================================
    
    """
    Show the confusion matrix. Use the same methods that were used in the
    first part of the lab (i.e., see Lab5.py)
    """
    # ====================== YOUR CODE HERE ======================

    # Get the predictions for the test set samples given by the classifier
    y_pred = knn.predict(X_test)

    # Calculate the confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Plot the confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
    # ============================================================
