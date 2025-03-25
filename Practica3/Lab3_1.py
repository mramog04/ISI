import numpy as np

Matrix1 = np.array([[4,-2,7],[9,4,1],[5,-1,5]])
print(Matrix1)

Matrix2 = Matrix1.transpose()
print(Matrix2)

result = Matrix1 * Matrix2
print(result)


prodM1M2 = Matrix1 @ Matrix2
print(prodM1M2)
prodM2M1 = Matrix2 @ Matrix1
print(prodM2M1)

mat_corners = np.array([[Matrix1[0,0],Matrix1[0,2]],[Matrix1[2,0],Matrix1[2,2]]])
print(mat_corners)


r1 = Matrix1[0,0];r2 = Matrix1[1,0];r3 = Matrix1[2,0]
for n in range(3):
    for m in range(3):
        if n == 0:
            if(Matrix1[n,m] > r1):
                r1 = Matrix1[n,m]
        elif n == 1:
            if(Matrix1[n,m] > r2):
                r2 = Matrix1[n,m]
        elif n == 2:
            if(Matrix1[n,m] > r3):
                r3 = Matrix1[n,m]
vec_max = np.array([r1,r2,r3])
print(vec_max)
max = vec_max.max()
print(max)

r1_m = Matrix1[0,0];r2_m = Matrix1[1,0];r3_m = Matrix1[2,0]
for n in range(3):
    for m in range(3):
        if n == 0:
            if(Matrix1[n,m] < r1_m):
                r1_m = Matrix1[n,m]
        elif n == 1:
            if(Matrix1[n,m] < r2_m):
                r2_m = Matrix1[n,m]
        elif n == 2:
            if(Matrix1[n,m] < r3_m):
                r3_m= Matrix1[n,m]
vec_min = np.array([r1_m,r2_m,r3_m])
min = vec_min.min()
print(vec_min)
print(min)

vec_min = vec_min.reshape(-1,1)
print(vec_min)

array = vec_min * vec_max
print(array)
array2 = vec_max * vec_min
print(array2)


print(Matrix1[0:3,0])
print(Matrix1[0:3,2])
mat_sum =Matrix1[0:3,0]+Matrix1[0:3,2]
print(mat_sum)



