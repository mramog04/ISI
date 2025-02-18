numeroDNI = input("Introduce tu número de DNI (sin la letra): ")
while True:
    try:
        numeroDNI = int(numeroDNI)
        if numeroDNI < 0 or numeroDNI > 99999999:
            print("Por favor, introduzca un número de DNI válido.")
        else:
            break
    except ValueError:
        print("Por favor, introduzca un número de DNI válido.")
    numeroDNI = input("Introduce tu número de DNI (sin la letra): ")
resto = numeroDNI % 23
listaLetras = ["T", "R", "W", "A", "G", "M", "Y", "F", "P", "D", "X", "B", "N", "J", "Z", "S", "Q", "V", "H", "L", "C", "K", "E"]

print("Tu letra de DNI es: ", listaLetras[resto])