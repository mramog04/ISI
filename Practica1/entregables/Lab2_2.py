listaNumeros = []
newNumero = input("Introduzca un número: ")
while True:
    try:
        newNumero = int(newNumero)
        if newNumero < 0:
            listaNumeros.append(newNumero)
            break
        listaNumeros.append(newNumero)
    except ValueError:
        print("Por favor, introduzca un número válido.")
    
    newNumero = input("Introduzca un número: ")
print(listaNumeros)
listaCuadrados = []
for numero in listaNumeros:
    listaCuadrados.append(numero ** 2)
print(listaCuadrados)
sumacuadrados = 0
for numero in listaCuadrados:
    sumacuadrados += numero
print(sumacuadrados)