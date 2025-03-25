import numpy as np

Matrix1 = np.random.uniform(0,3,(20,20))
print(Matrix1)

pos_list = []

for n in range(0,Matrix1.shape[0]):
    for m in range(0,Matrix1.shape[1]):
        if Matrix1[n,m] >= 1 and Matrix1[n,m] <= 2:
            pos_list.append((n,m))
            
print(pos_list)
print(len(pos_list))

pos_list2 = []

for n in range(0,Matrix1.shape[0]):
    for m in range(0,Matrix1.shape[1]):
        if Matrix1[n,m] <= 1 or Matrix1[n,m] >= 2:
            pos_list2.append((n,m))
            
            
""" x = np.nonzero(Matrix1 <= 1 or Matrix1 >= 2) """

print(x)
print(len(x))
            
print(pos_list2)
print(len(pos_list2))

for n in range(0,Matrix1.shape[0]):
    for m in range(0,Matrix1.shape[1]):
        Matrix1[n,m] = round(Matrix1[n,m], 0)
        


print(Matrix1)

pos_list3 = []

for n in range(0,Matrix1.shape[0]):
    for m in range(0,Matrix1.shape[1]):
        if Matrix1[n,m] != 1:
            pos_list3.append((n,m))
print(pos_list3)
print(len(pos_list3))  

