numero = input("Introduzca un número para calcular su factorial: ")

while True:
    try:
        numero = int(numero)
        if numero < 0:
            print("El numero es negativo, se trabajara con su valor absoluto.")
            numero = abs(numero)
            break
        else:
            break
    except ValueError:
        print("Por favor, introduzca un número válido.")
    numero = input("Introduzca un número para calcular su factorial: ")
    
aux = numero
for i in range(1, numero):
    numero = numero * i
print("El factorial de ",aux," es: ", numero)
