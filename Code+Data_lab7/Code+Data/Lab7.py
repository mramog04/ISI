#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Creado el Mar Oct 25 12:19:51 2022
Modificado en Marzo 2025

@author: TU NOMBRE AQUÍ
"""

import h5py
import os
import numpy as np
from sklearn import preprocessing
from sklearn import svm
from sklearn.metrics import confusion_matrix


def crear_k_folds(n_splits, clases, estratificado=True):
    """
    Crea un vector con tantos elementos como elementos hay en el conjunto de datos. Cada
    elemento de dicho vector contiene el número del índice del fold en
    el que debe estar ese elemento. Esta asignación se hace de forma aleatoria.
    
    Parámetros
        ----------
        n_splits: int
            Número de folds a generar.
        clases: numpy 1D array
            Vector que indica las clases de los elementos del conjunto de datos.
        estratificado: boolean
            Variable booleana que indica si la partición k-fold
            debe ser estratificada (True) o no (False). Valor por defecto: False

    Retornos
        -------
        indices_folds: numpy 1D array
            Vector (numpy 1D array) con la misma longitud que el vector de entrada y
            que contiene el fold en el que debe estar el elemento correspondiente del
            conjunto de datos.
            Esto significa que, si la posición i-ésima del vector de salida es N, entonces
            el elemento X[i] del conjunto de datos, cuya clase es y[i], estará en el
            fold N.
    """
    indices_folds = np.zeros(clases.shape, dtype=int)

    if not estratificado:
        aux = 0
        # ====================== TU CÓDIGO AQUÍ ======================
        for j in range(indices_folds.size):
            indices_folds[j] = aux
            aux += 1
            if aux == n_splits:
                aux = 0
        np.random.shuffle(indices_folds)
        # ============================================================

    else:
        # ====================== TU CÓDIGO AQUÍ ======================
        clases_0 = np.where(clases == 0)#indices de la clase 0 en classes
        clases_1 = np.where(clases == 1)#indices de la clase 1 en classes
        clases_0 = np.array(clases_0).flatten()
        clases_1 = np.array(clases_1).flatten()
        contador = 0
        aux1 = len(clases_0)-1
        aux2 = len(clases_1)-1
        for i in range(indices_folds.size):
            if contador == n_splits:
                contador = 0
            if aux1 >= 0:
                indices_folds[clases_0[aux1]] = contador
                aux1 -= 1
            else:
                indices_folds[clases_1[aux2]] = contador
                aux2 -= 1
            contador += 1
        """ np.random.shuffle(indices_folds) """
        
        # ============================================================

    return indices_folds


# %%
# -------------
# PROGRAMA PRINCIPAL
# -------------
if __name__ == "__main__":

    np.random.seed(67)

    dir_data = "Data"
    data_path = os.path.join(dir_data, "mammographic_data.h5")
    
# %%
# -------------
# PRELIMINAR: CARGAR CONJUNTO DE DATOS
# -------------

    # Importar datos del csv usando pandas
    '''
    mammographic_data_df = pd.read_csv(data_path)
    y_df = mammographic_data_df[['Class']].copy()
    X_df = mammographic_data_df.copy()
    X_df = X_df.drop('Class', axis=1)

    X = X_df.to_numpy()
    y = y_df.to_numpy().flatten()

    # Importar datos del archivo h5 (EN CASO DE QUE LA IMPORTACIÓN DESDE EL CSV NO FUNCIONE CORRECTAMENTE)
    '''
    # importar características y etiquetas
    h5f_data = h5py.File("mammographic_data.h5", 'r')

    features_ds = h5f_data['data']
    labels_ds = h5f_data['labels']

    X = np.array(features_ds)
    y = np.array(labels_ds).flatten()

    h5f_data.close()

# %%
# -------------
# PARTE 1: CREAR K FOLDS Y VERIFICAR LAS PROPORCIONES
# -------------
    K = 10  # número de folds

    # Generar los índices de los folds llamando a crear_k_folds
    # ====================== TU CÓDIGO AQUÍ ======================
    idx_folds = crear_k_folds(K, y)
    # ============================================================
    
    proporcion_clase_0 = np.sum(y == 0) / y.size
    proporcion_clase_1 = 1 - proporcion_clase_0
    print("**********************************************************")
    print("****** VERIFICA LAS PROPORCIONES DE LAS CLASES DENTRO DE LOS FOLDS ******")
    print("**********************************************************")
    print("\n")
    print("La distribución del conjunto de datos completo es:")
    print("- {:.2f} % elementos de la clase 0".format(
        100 * proporcion_clase_0))
    print("- {:.2f} % elementos de la clase 1".format(
        100 * proporcion_clase_1))
    print("\n")
    print("La distribución de los elementos dentro de cada fold es:")

    for i in range(K):
        # Obtener los índices de los elementos del conjunto de prueba (es decir, aquellos en el fold i)
        test_index = np.nonzero(idx_folds == i)[0]
        # Obtener los índices de los elementos del conjunto de entrenamiento (es decir, aquellos en los otros folds)
        train_index = np.nonzero(idx_folds != i)[0]

        prop_clase_0_train = np.sum(y[train_index] == 0) / train_index.size
        prop_clase_1_train = 1 - prop_clase_0_train
        prop_clase_0_test = np.sum(y[test_index] == 0) / test_index.size
        prop_clase_1_test = 1 - prop_clase_0_test
        print("* FOLD {}:".format(i+1))
        print("  - TRAIN: {:.2f} % elementos de la clase 0;  {:.2f} % elementos de la clase 1".format(
              100 * prop_clase_0_train, 100 * prop_clase_1_train))
        print("  - TEST: {:.2f} % elementos de la clase 0;  {:.2f} % elementos de la clase 1".format(
              100 * prop_clase_0_test, 100 * prop_clase_1_test))

# %%
# -------------
# PARTE 2: VALIDACIÓN CRUZADA CON SVM
# -------------

    # Parámetros para SVM
    valor_C = 1
    tipo_kernel = "linear" # Debes probar diferentes kernels. Lee la documentación

    # Inicialización de los vectores para almacenar las precisiones y Fscores
    # de cada fold
    precisiones = np.zeros(shape=(K,))
    Fscores = np.zeros(shape=(K,))

    # Proceso iterativo de validación cruzada
    for i in range(K):
        # Utiliza los índices de los elementos de entrenamiento y prueba del i-ésimo fold
        # para extraer los subconjuntos de entrenamiento y prueba de este fold.
        # ====================== TU CÓDIGO AQUÍ ======================
        X_train_fold = X[np.nonzero(idx_folds != i)[0]]
        y_train_fold = y[np.nonzero(idx_folds != i)[0]]
        X_test_fold = X[np.nonzero(idx_folds == i)[0]]
        y_test_fold = y[np.nonzero(idx_folds == i)[0]]
        # ============================================================

        # Estandarizar datos de este fold
        scaler = preprocessing.StandardScaler()
        scaler.fit(X_train_fold)
        X_train_fold = scaler.transform(X_train_fold)
        X_test_fold = scaler.transform(X_test_fold)

        # Instanciar el SVM con el tipo de kernel y valor C definidos, 
        # entrenarlo y usarlo para clasificar. Usa los conjuntos de entrenamiento
        # y prueba de la iteración actual.
        # ====================== TU CÓDIGO AQUÍ ======================
        # Instanciar
        svm_clf = svm.SVC(kernel=tipo_kernel, C=valor_C)
        # Entrenar
        svm_clf.fit(X_train_fold, y_train_fold)
        
        # Clasificar conjunto de prueba
        y_test_assig_fold = svm_clf.predict(X_test_fold)
        confm = confusion_matrix(y_true=y_test_fold, y_pred=y_test_assig_fold)
        # ============================================================

        # Calcular la precisión y f-score del conjunto de prueba en este fold y
        # almacenarlos en los vectores de precisiones y Fscores, respectivamente
        # ====================== TU CÓDIGO AQUÍ ======================
        precision_fold = np.mean(y_test_fold == y_test_assig_fold)
        TP = confm[1,1]
        FP = confm[0,1]
        FN = confm[1,0]
        Fscore_fold = 2*TP/(2*TP+FP+FN)
        
        precisiones[i] = precision_fold
        Fscores[i] = Fscore_fold
        # ============================================================


# %%
# -------------
# PARTE 3: MOSTRAR RESULTADOS FINALES
# -------------

    print("\n\n")
    print('***********************************************')
    print('******* RESULTADOS DE LA VALIDACIÓN CRUZADA *******')
    print('***********************************************')
    print('\n')

    for i in range(K):
        print("FOLD {}:".format(i+1))
        print("    Precisión = {:4.3f}".format(precisiones[i]))
        print("    Fscore = {:5.3f}".format(Fscores[i]))

    # ====================== TU CÓDIGO AQUÍ ======================
    # Calcular la media y la desviación estándar de las precisiones y F1-scores
    media_precision = np.mean(precisiones)
    std_precision = precisiones.std()
    media_fscore = np.mean(Fscores)
    std_fscore = Fscores.std()
    # ============================================================

    print("\n")
    print("PRECISIÓN PROMEDIO = {:4.3f}; DESVIACIÓN ESTÁNDAR DE LA PRECISIÓN = {:4.3f}".format(
        media_precision, std_precision))
    print("FSCORE PROMEDIO = {:4.3f}; DESVIACIÓN ESTÁNDAR DEL FSCORE = {:4.3f}".format(
        media_fscore, std_fscore))
    print("\n")
    print('***********************************************')
    print('***********************************************')
    print('***********************************************')
    print("\n\n\n")