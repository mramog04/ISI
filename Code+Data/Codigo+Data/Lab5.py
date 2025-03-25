#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Creado el Viernes 14 de Octubre de 2022
Modificado el Lunes 13 de Marzo de 2023
Modificado el Jueves 13 de Marzo de 2025

@author: CAMBIAR EL NOMBRE
"""

# Importar las bibliotecas necesarias
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def classify_kNN(X_train, y_train, X_test, k):
    """
    Esta función implementa el algoritmo de clasificación kNN con la
    distancia euclidiana

    Parámetros
    ----------
    X_train : ndarray
        Matriz (n_Train x m), donde n_Train es el número de elementos de entrenamiento
        y m es el número de características (longitud del vector de características).
    y_train : ndarray
        Las clases de los elementos en el conjunto de entrenamiento. Es un
        vector de longitud n_Train con el número de la clase.
    X_test : ndarray
        Matriz (n_t x m), donde n_t es el número de elementos en el conjunto de prueba.
    k : int
        Número de vecinos más cercanos a considerar para hacer una asignación.

    Retorna
    -------
    y_test_assig : ndarray
        Un vector de longitud n_t, con las clases asignadas por el algoritmo
        a los elementos en el conjunto de prueba.
    """

    num_elements_test = X_test.shape[0]
    
    # Reservar espacio para el vector de salida con las asignaciones
    y_test_assig = np.empty(shape=(num_elements_test, 1), dtype=int)
    distances_array = np.zeros((num_elements_test, X_train.shape[0]))
    k_selected = np.zeros(k,dtype=int)

    # Para cada elemento en el conjunto de prueba...
    for i in range(num_elements_test):
        """
        1 - Calcular la distancia euclidiana del elemento i del conjunto de prueba
        a todos los elementos del conjunto de entrenamiento
        """
        # ====================== TU CÓDIGO AQUÍ ======================
        for j in range(X_train.shape[0]):
            distances_array[i][j]=np.linalg.norm(X_test[i]-X_train[j])
            
        # ============================================================

        """
        2 - Ordenar las distancias en orden ascendente y utilizar los índices del ordenamiento
        """
        # ====================== TU CÓDIGO AQUÍ ======================
        distances_sorted = np.argsort(distances_array[i])
        # ============================================================

        """
        3 - Tomar las k primeras clases del conjunto de entrenamiento
        """
        # ====================== TU CÓDIGO AQUÍ ======================
        for j in range(k):
            k_selected[j] = y_train[distances_sorted[j]]
        # ============================================================

        """
        4 - Asignar al elemento i la clase más frecuente
        """
        # ====================== TU CÓDIGO AQUÍ ======================
        y_test_assig[i] = np.bincount(k_selected).argmax()
        # ============================================================
    return y_test_assig


# -------------
# PROGRAMA PRINCIPAL
# -------------
if __name__ == "__main__":

    # PARTE 1: CARGAR CONJUNTO DE DATOS Y DIVISIÓN ENTRE ENTRENAMIENTO Y PRUEBA

    # Cargar archivo CSV con los datos en un DataFrame de pandas
    """
    Cada fila en este DataFrame contiene un vector de características, que representa un
    correo electrónico.
    Cada columna representa una variable, EXCEPTO LA ÚLTIMA COLUMNA, que representa
    la clase real del elemento correspondiente (es decir, la fila): 1 significa "spam",
    y 0 significa "no spam"
    """
    dir_data = "Data"
    spam_df = pd.read_csv("spambase_data.csv")
    y_df = spam_df[['Class']].copy()
    X_df = spam_df.copy()
    X_df = X_df.drop('Class', axis=1)

    # Convertir DataFrame a un array de numpy
    X = X_df.to_numpy()
    y = y_df.to_numpy()

    """
    Número de elementos del conjunto de datos y número de variables de cada vector de características
    que representa cada correo spam
    """
    num_elements, num_variables = X.shape

    """
    Parámetro que indica la proporción de elementos que tendrá el conjunto de prueba
    """
    proportion_test = 0.3

    """
    En la siguiente sección se realiza la partición del conjunto de datos en conjuntos de entrenamiento y prueba.
    Observa los resultados producidos por cada línea de código para comprender qué hace,
    usando el depurador si es necesario.
    
    Luego, escribe una breve explicación para cada línea con comentarios.
    """
    # ============================================
    num_elements_train = int(num_elements * (1 - proportion_test))

    inds_permutation = np.random.permutation(num_elements)

    inds_train = inds_permutation[:num_elements_train]
    inds_test = inds_permutation[num_elements_train:]

    X_train = X[inds_train, :]
    y_train = y[inds_train]

    X_test = X[inds_test, :]
    y_test = y[inds_test]
    # ============================================

    # ***********************************************************************
    # ***********************************************************************
    # PARTE 2: ALGORITMO DE K-VECINOS MÁS CERCANOS (K-NEAREST NEIGHBOURS)

    k = 3
    """
    La función classify_kNN implementa el algoritmo kNN. Revísala y
    completa el código
    """
    y_test_assig = classify_kNN(X_train, y_train, X_test, k)

    # ***********************************************************************
    # ***********************************************************************
    # PARTE 3: EVALUACIÓN DEL DESEMPEÑO DEL CLASIFICADOR

    # Mostrar matriz de confusión
    confusion_matrix_kNN = confusion_matrix(y_true=y_test, y_pred=y_test_assig)
    print(confusion_matrix_kNN)

    # Si deseas imprimir la matriz de confusión usando matplotlib
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_kNN)
    disp.plot()
    plt.title('Matriz de Confusión', fontsize=14)
    plt.show()

    # Precisión: Proporción de elementos bien clasificados entre todos los elementos
    # ====================== TU CÓDIGO AQUÍ ======================
    accuracy = np.mean(y_test_assig == y_test)
    # ============================================================
    print('Precisión: {:.4f}'.format(accuracy))

    # Sensibilidad: Proporción de elementos bien clasificados entre los positivos reales
    # ====================== TU CÓDIGO AQUÍ ======================
    sensitivity = confusion_matrix_kNN[1, 1] / (confusion_matrix_kNN[1, 0] + confusion_matrix_kNN[1, 1])
    # ============================================================
    print('Sensibilidad (TPR): {:.4f}'.format(sensitivity))

    # Especificidad: Proporción de elementos bien clasificados entre los negativos reales
    # ====================== TU CÓDIGO AQUÍ ======================
    specificity = confusion_matrix_kNN[0, 0] / (confusion_matrix_kNN[0, 0] + confusion_matrix_kNN[0, 1])
    # ============================================================
    print('Especificidad (TNR): {:.4f}'.format(specificity))
