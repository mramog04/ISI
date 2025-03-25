#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Creado el Lunes 25 de octubre de 2022
Modificado el 20 de marzo de 2025

@autor: TU NOMBRE AQUÍ

"""

import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
import numpy as np
import h5py
import matplotlib.pyplot as plt
from math import exp, log


def calculate_cost_log_reg(y, y_hat):
    """
    Calcula el costo de la SALIDA DE LOS PATRONES DE TODO EL CONJUNTO DE ENTRENAMIENTO
    del clasificador de regresión logística (es decir, el resultado de aplicar la función h
    a todos los patrones del conjunto de entrenamiento) y sus clases reales.

    Parámetros
        ----------
        y: vector numpy
            Vector con las clases reales del conjunto de entrenamiento.
        y_hat: vector numpy
            Salida de la función h (es decir, la hipótesis del clasificador
            de regresión logística para cada elemento del conjunto de entrenamiento).
         ----------

    Retornos
        -------
        cost_set: float
            Valor del costo de las salidas estimadas del conjunto de entrenamiento.
        -------
    """

    # ====================== TU CÓDIGO AQUÍ ======================
    m = len(y)
    epsilon = 1e-15
    y_hat = np.clip(y_hat, epsilon, 1-epsilon)
    cost_i = 0
    for i in range(m):
        cost_i += y[i] * log(y_hat[i]) + (1-y[i]) * log(1-y_hat[i])
    # ============================================================
    cost_i = -cost_i/m

    return cost_i


# **************************************************************************
# **************************************************************************
def fun_sigmoid(theta_sigmoid, x):
    """
    Esta función calcula la función sigmoide g(z), donde z es una combinación
    lineal de los parámetros theta y los componentes del vector de características x.

    Parámetros
        ----------
        theta_sigmoid: vector numpy
            Parámetros de la función h del clasificador de regresión logística.
        x: vector numpy
            Vector que contiene los datos de un patrón.
         ----------

    Retornos
        -------
        g: float
            Resultado de aplicar la función sigmoide usando la combinación
            lineal de theta y X.
        -------
    """
    # ====================== TU CÓDIGO AQUÍ ======================
    z = np.dot(theta_sigmoid, x)
    g = 1/(1+np.exp(-z))
    # ============================================================

    return g


# **************************************************************************
# **************************************************************************
def train_logistic_regression(X_train, y_train, alpha_train,
                              verbose=True, max_iter=100):
    """
    Esta función implementa el entrenamiento de un clasificador de regresión
    logística usando los datos de entrenamiento (X_train) y sus clases (y_train).

    Parámetros
        ----------
        X_train: matriz numpy
            Matriz con dimensiones (m x n) con los datos de entrenamiento, donde m es
            el número de patrones de entrenamiento (es decir, elementos) y n es el número
            de características (es decir, la longitud del vector de características que
            caracteriza al objeto).
        y_train: vector numpy
            Vector que contiene las clases de los patrones de entrenamiento. Su
            longitud es n.
        alpha_train: float
            Escalar que contiene la tasa de aprendizaje.

    Retornos
        -------
        theta: vector numpy
            Vector con longitud n (es decir, la misma longitud que el número de
            características en cada patrón). Contiene los parámetros theta de la
            función de hipótesis obtenidos después del entrenamiento.

    """

    # Número de patrones de entrenamiento.
    m = np.shape(X_train)[0]

    # Asignar espacio para las salidas de la función de hipótesis para cada
    # patrón de entrenamiento
    h_train = np.zeros(shape=m)

    # Asignar espacios para los valores de la función de costo en cada iteración
    cost_values = np.zeros(shape=(1 + max_iter))

    # Inicializar el vector para almacenar los parámetros de la función de hipótesis
    # Todos los valores en la inicialización son cero
    # heta_train = np.zeros(shape=(1, 1 + np.shape(x_train)[1]))  -> todos ceros
    # Todos los valores en la inicialización caen dentro del intervalo [a, b)
    a = -10
    b = 10
    theta_train = np.random.uniform(low=a,
                                    high=b,
                                    size=(1 + np.shape(X_train)[1]))

    # -------------
    # CALCULAR EL VALOR DE LA FUNCIÓN DE COSTO PARA LOS THETAS INICIALES
    # -------------
    # a. Resultado intermedio: Obtener la estimación (es decir, salida de la
    # regresión logística) para cada elemento
    for i in range(m):
        # Añadir un 1 (es decir, el valor para x0) al principio de cada patrón
        x_i = np.insert(np.array([X_train[i]]), 0, 1)

        # Salida esperada (es decir, resultado de la función sigmoide) para el
        # patrón i-ésimo, y almacenarlo en h_train para uso futuro
        # ====================== TU CÓDIGO AQUÍ ======================
        h_train[i]=fun_sigmoid(theta_train, x_i)
        # ============================================================

    # b. Calcular el costo
    # ====================== TU CÓDIGO AQUÍ ======================
    cost_values[0] = calculate_cost_log_reg(y_train, h_train)
    # ============================================================

    # -------------
    # ALGORITMO DE DESCENSO DE GRADIENTE PARA ACTUALIZAR LOS THETAS
    # -------------
    # Método iterativo llevado a cabo durante un número máximo (max_iter) de
    # iteraciones
    for num_iter in range(max_iter):

        # ------
        # PASO 1. Actualizar los valores de theta. Para hacerlo, sigue las
        # ecuaciones de actualización estudiadas en las sesiones teóricas.
        #
        # RECUERDA QUE, EN ESTE PUNTO DEL BUCLE, LAS ESTIMACIONES DADAS POR
        # LA FUNCIÓN SIGMOIDE YA ESTÁN CALCULADAS Y ALMACENADAS EN h_train.
        # 
        # a. Resultado intermedio: Calcular el (h_i-y_i)*x para CADA elemento
        # del conjunto de entrenamiento
        aux = 0
        theta_old = np.copy(theta_train)
        for i in range(m):
            # Añadir un 1 (es decir, el valor para x0) al principio de cada patrón
            x_i = np.insert(np.array([X_train[i]]), 0, 1)

            # ====================== TU CÓDIGO AQUÍ ======================
            aux += (h_train[i] - y_train[i]) * x_i
            """ theta_train[i] = theta_old[i] - alpha_train * aux """
            # ============================================================

        # b. Actualización de los thetas
        # ====================== TU CÓDIGO AQUÍ ======================
        theta_train = theta_old - alpha_train * aux
        # ============================================================
        
        
        # ------
        # PASO 2: Calcular el costo en esta iteración y almacenarlo en el
        # vector cost_values.
        # a. Resultado intermedio: Obtener el error para cada elemento y sumarlo.
        for i in range(m):
            # Añadir un 1 (es decir, el valor para x0) al principio de cada patrón
            x_i = np.insert(np.array([X_train[i]]), 0, 1)

            # Salida esperada (es decir, resultado de la función sigmoide) para el
            # patrón i-ésimo, y almacenarlo en h_train para uso futuro
            # ====================== TU CÓDIGO AQUÍ ======================
            h_train[i] = fun_sigmoid(theta_train, x_i)
            # ============================================================

        # b. Calcular el costo
        # ====================== TU CÓDIGO AQUÍ ======================
        cost_values[num_iter+1] = calculate_cost_log_reg(y_train, h_train)
        # ============================================================

        '''
        CRITERIO DE PARADA TEMPRANA: Si el valor absoluto del costo en la iteración actual
        con respecto a la iteración anterior es menor que 0.001,
        detener el entrenamiento.
        '''
        # ====================== TU CÓDIGO AQUÍ ======================
        if(abs(cost_values[num_iter+1]-cost_values[num_iter])<0.001):
            break
        # ============================================================


    # Si verbose es True, graficar el costo en función del número de iteraciones
    if verbose:
        plt.plot(cost_values, color='red')
        plt.xlabel('Número de iteraciones')
        plt.ylabel('Costo J')
        plt.title(r'Función de costo a lo largo de las iteraciones con $\alpha=${}'.
                  format(alpha), fontsize=14)
        plt.show()

    return theta_train


# **************************************************************************
# **************************************************************************
def classify_logistic_regression(X_test, theta_classif):
    """
    Esta función devuelve la probabilidad para cada patrón del conjunto de prueba
    de pertenecer a la clase positiva usando el clasificador de regresión logística.

    Parámetros
        ----------
        X_test: matriz numpy
            Matriz con dimensiones (m_t x n) con los datos de prueba, donde m_t
            es el número de patrones de prueba y n es el número de características
            (es decir, la longitud del vector de características que define cada elemento).
        theta_classif: vector numpy
            Parámetros de la función h del clasificador de regresión logística.

    Retornos
        -------
        y_hat: vector numpy
            Vector de longitud m_t con las estimaciones realizadas para cada elemento
            de prueba mediante el clasificador de regresión logística. Estas
            estimaciones corresponden a las probabilidades de que estos elementos
            pertenezcan a la clase positiva.
    """

    num_elem_test = np.shape(X_test)[0]
    y_hat = np.zeros(shape=(num_elem_test, 1))

    for i in range(num_elem_test):
        # Añadir un 1 (valor para x0) al principio de cada patrón
        x_test_i = np.insert(np.array([X_test[i]]), 0, 1)
        # ====================== TU CÓDIGO AQUÍ ======================
        y_hat[i] = fun_sigmoid(theta_classif, x_test_i)
        # ============================================================

    return y_hat


# **************************************************************************
# **************************************************************************
# -------------
# PROGRAMA PRINCIPAL
# -------------
if __name__ == "__main__":

    np.random.seed(325)
    
    plt.close('todo')

    dir_data = "Datos"
    data_path = os.path.join(dir_data, "mammographic_data.h5")
    test_size = 0.3
    decision_threshold = 0.5

    # -------------
    # PRELIMINAR: CARGAR EL CONJUNTO DE DATOS Y DIVIDIR LOS CONJUNTOS DE ENTRENAMIENTO Y PRUEBA (NO NECESITA
    # CAMBIAR NADA)
    # -------------

    # Importar datos del archivo csv
    '''
    # Importar datos del csv usando pandas
    mammographic_data_df = pd.read_csv(data_path)
    y_df = mammographic_data_df[['Class']].copy()
    X_df = mammographic_data_df.copy()
    X_df = X_df.drop('Class', axis=1)

    X = X_df.to_numpy()
    y = y_df.to_numpy().flatten()
    '''

    # Importar datos del archivo h5
    # importar características y etiquetas
    h5f_data = h5py.File("mammographic_data.h5", 'r')

    features_ds = h5f_data['data']
    labels_ds = h5f_data['labels']

    X = np.array(features_ds)
    y = np.array(labels_ds).flatten()

    h5f_data.close()

    # DIVIDIR LOS DATOS EN CONJUNTOS DE ENTRENAMIENTO Y PRUEBA
    # ====================== TU CÓDIGO AQUÍ ======================
    num_elements, num_variables = X.shape
    num_elements_train = int(num_elements * (1 - test_size))

    inds_permutation = np.random.permutation(num_elements)

    inds_train = inds_permutation[:num_elements_train]
    inds_test = inds_permutation[num_elements_train:]

    X_train = X[inds_train, :]
    y_train = y[inds_train]

    X_test = X[inds_test, :]
    y_test = y[inds_test]
    # ============================================================

    # ESTANDARIZAR LOS DATOS
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # print("Media del conjunto de entrenamiento: {}".format(X_train.mean(axis=0)))
    # print("Desviación estándar del conjunto de entrenamiento: {}".format(X_train.std(axis=0)))
    # print("Media del conjunto de prueba: {}".format(X_test.mean(axis=0)))
    # print("Desviación estándar del conjunto de prueba: {}".format(X_test.std(axis=0)))

    # -------------
    # PARTE 1: ENTRENAMIENTO DEL CLASIFICADOR Y CLASIFICACIÓN DEL CONJUNTO DE PRUEBA
    # -------------

    # ENTRENAMIENTO

    # Tasa de aprendizaje. Cambiarla según sea necesario, dependiendo de cómo
    # evoluciona la función de costo a lo largo de las iteraciones.
    alpha = 2

    # La función train_logistic_regression implementa el clasificador de regresión
    # logística. Ábrela y completa el código.
    theta = train_logistic_regression(X_train, y_train, alpha)

    # print(theta)

    # -------------
    # CLASIFICACIÓN DEL CONJUNTO DE PRUEBA
    # -------------
    y_test_hat = classify_logistic_regression(X_test, theta)

    # Asignación de la clase
    y_test_assig = y_test_hat >= decision_threshold

    # -------------
    # PARTE 2: DESEMPEÑO DEL CLASIFICADOR: CÁLCULO DE LA PRECISIÓN Y PUNTUACIÓN F
    # -------------

    # Mostrar matriz de confusión
    # y_test = np.array([y_test.astype(bool)])
    # confm = confusion_matrix(y_test.T, y_test_assig.T)
    confm = confusion_matrix(y_true=y_test, y_pred=y_test_assig)
    print(confm)
    # classNames = np.arange(0,1)
    # disp = ConfusionMatrixDisplay(confusion_matrix=confm,display_labels=classNames)
    disp = ConfusionMatrixDisplay(confusion_matrix=confm)
    disp.plot()
    plt.title('Matriz de Confusión', fontsize=14)
    plt.show()

    # -------------
    # PARTE 3: PRECISIÓN Y PUNTUACIÓN F
    # -------------

    # Precisión
    # ====================== TU CÓDIGO AQUÍ ======================
    accuracy = np.mean(y_test_assig == y_test)
    # ============================================================
    print("***************")
    print("La precisión del clasificador de Regresión Logística es {:.4f}".
          format(accuracy))
    print("***************")

    # Puntuación F1
    # ====================== TU CÓDIGO AQUÍ ======================
    TP = confm[1, 1]
    FP = confm[0, 1]
    FN = confm[1, 0]
    f_score = 2*TP/(2*TP+FP+FN)
    # ============================================================
    print("")
    print("***************")
    print("La puntuación F1 del clasificador de Regresión Logística es {:.4f}".
          format(f_score))
    print("***************")