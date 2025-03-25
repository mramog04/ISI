import numpy as np
import timeit


def numpyMatrices(n):
    Matrix1 = np.zeros(int(n))
    for i in range(int(n)):
        Matrix1[i] = i**2
    print(Matrix1)
    return Matrix1

def calculateList(n):
    list = []
    for i in range(int(n)):
        list.append(i**2)
    print(list)
    return list

def vectorizeng(MatrixA,MatrixB):
    return MatrixA*MatrixB




n = input("Introduce un número: ")
n = int(n)
m1 = np.arange(1, n)

time_list = timeit.timeit("calculateList(n)", "from __main__ import calculateList, n", number=10000)
time_numpy = timeit.timeit("numpyMatrices(n)", "from __main__ import numpyMatrices, n", number=10000)
time_vector = timeit.timeit("vectorizeng(m1,m1)", "from __main__ import vectorizeng, m1", number=10000)

print("Tiempo de ejecución de la función calculateList: ", time_list)
print("Tiempo de ejecución de la función numpyMatrices: ", time_numpy)
print("Tiempo de ejecución de la función vectorizeng: ", time_vector)


