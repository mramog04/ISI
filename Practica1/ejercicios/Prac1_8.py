squares = [1, 4, 9, 16, 25]
print(squares)
print(len(squares))
print(squares[0])
print(squares[-1])
""" Esto imprime el primer y último elemento de la lista """
print(squares[:])
print(squares + [36, 49, 64, 81, 100])
""" Esto imprime la lista squares concatenada con otra lista """
squares = squares + [36, 49, 64, 81, 100]
print(squares[0:9:2])
""" Esto imprime los elementos de la lista squares desde el primer elemento hasta el cu"""

del squares[5:10]
print(squares)