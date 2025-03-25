import numpy as np

Matrix1 = np.random.randint(-10,10,(10,4))

print(Matrix1)

Matrix2 = np.zeros((10,10))

for n in range(0,Matrix1.shape[0]):
    for m in range(0,Matrix1.shape[0]):
        if n != m:
            diff = Matrix1[n,:] - Matrix1[m,:]
            Matrix2[n,m] = np.sqrt(np.sum(diff**2))
            
Matrixsave = Matrix2
Matrix3 = Matrix2
print(Matrix2)

for n in range(0,Matrix2.shape[0]):
    for m in range(0,Matrix2.shape[1]):
        Matrix3[n,m] = round(Matrix2[n,m], 0)
        
print(Matrix3)

for n in range(0,Matrix1.shape[0]):
    for m in range(0,Matrix1.shape[0]):
        print("La distancia euclidea entre el vector ",n," y el vector ",m," es: ",Matrix2[n,m])