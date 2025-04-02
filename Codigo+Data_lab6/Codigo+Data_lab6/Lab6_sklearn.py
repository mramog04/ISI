#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 23:24:10 2023
Modified on Mar 20 2025

@author: YOUR NAME HERE
"""

import h5py
import os
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt

# -------------
# MAIN PROGRAM
# -------------
if __name__ == "__main__":

    np.random.seed(325)
    plt.close('all')

    dir_data = "Data"
    data_path = os.path.join(dir_data, "mammographic_data.h5")
    test_size = 0.3
    decision_threshold = 0.5

    # -------------
    # PRELIMINARY: LOAD DATASET AND PARTITION TRAIN-TEST SETS (NO NEED TO
    # CHANGE ANYTHING)
    # -------------

    # import features and labels
    h5f_data = h5py.File("mammographic_data.h5", 'r')

    features_ds = h5f_data['data']
    labels_ds = h5f_data['labels']

    X = np.array(features_ds)
    y = np.array(labels_ds).flatten()

    h5f_data.close()

    # SPLIT DATA INTO TRAINING AND TEST SETS
    # ====================== YOUR CODE HERE ======================
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    # ============================================================

    # STANDARDIZE DATA
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # -------------
    # CLASSIFICATION WITH SCIKIT-LEARN'S LOGISTIC REGRESSION
    # -------------
    # Create an instance of the logistic regression classifier
    # ====================== YOUR CODE HERE ======================
    clf = LogisticRegression()
    # ============================================================

    # Train the classifier
    # ====================== YOUR CODE HERE ======================
    clf.fit(X_train, y_train)
    # ============================================================

    # Predict the classes of the test set samples using the trained classifier
    # ====================== YOUR CODE HERE ======================
    y_test_assig_skl = clf.predict(X_test)
    # ============================================================

    # OTHER WAY OF PREDICTING THE CLASSES: Get the soft outputs (i.e. the
    # probabilities) of the test set and compare them with decision_threshold
    # ====================== YOUR CODE HERE ======================
    y_test_probs_1 = clf.predict_proba(X_test)[:, 1]
    y_test_probs_0 = clf.predict_proba(X_test)[:, 0]
    print(y_test_probs_0)
    print(y_test_probs_1)
    # ============================================================

    # Display confusion matrix
    confm_skl = confusion_matrix(y_true=y_test, y_pred=y_test_assig_skl)
    disp = ConfusionMatrixDisplay(confusion_matrix=confm_skl)
    disp.plot()
    plt.title('Confusion Matrix for the scikit-learn classifier', fontsize=14)
    plt.show()

    
      
    # -------------
    # ACCURACY AND F-SCORE
    # -------------

    # Accuracy
    # ====================== YOUR CODE HERE ======================
    accuracy = np.mean(y_test_assig_skl == y_test)
    # ============================================================
    print("***************")
    print("The accuracy of the Logistic Regression classifier is {:.4f}".
          format(accuracy))
    print("***************")

    # F1 score
    # ====================== YOUR CODE HERE ======================
    TP = confm_skl[1, 1]
    FP = confm_skl[0, 1]
    FN = confm_skl[1, 0]
    f_score = 2*TP/(2*TP+FP+FN)
    # ============================================================
    print("")
    print("***************")
    print("The F1-score of the Logistic Regression classifier is {:.4f}".
          format(f_score))
    print("***************")
