import random
from time import sleep


def calc_puntos(n,fallos):
    return n/(2**fallos)

def ver_puntuacion():
    archivo = open("puntuaciones.txt", "r")
    print(archivo.read())
    archivo.close()
    menu()

def menu():
    print("1. Jugar")
    print("2. Ver puntuación")
    print("3. Salir")
    opcion = input("Introduce una opción: ")
    while True:
        try:
            opcion = int(opcion)
            if opcion < 1 or opcion > 3:
                print("Por favor, introduzca una opción válida.")
            else:
                break
        except ValueError:
            print("Por favor, introduzca una opción válida.")
        opcion = input("Introduce una opción: ")
    if(opcion == 1):
        jugar()
    elif(opcion == 2):
        ver_puntuacion()
    elif(opcion == 3):
        print("Gracias por jugar.")
        exit()

def jugar():
    archivo = open("puntuaciones.txt", "a")
    archivo.write("\n")
    nombre = input("Introduce tu nombre: ")
    
    while True:
        try:
            if nombre == "":
                print("Por favor, introduzca un nombre válido.")
            else:
                break
        except ValueError:
            print("Por favor, introduzca un nombre válido.")
        nombre = input("Introduce tu nombre: ")
    
    archivo.write("Nuevo jugador: " + nombre + "\n")
    numero = input("Introduzca un numero entero para comenzar el juego:")
    
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
        numero = input("Introduzca un número entero para comenzar el juego: ")
    
    numeroAleatorio = random.randint(1, numero)
    print("Se ha generado un número aleatorio entre 1 y ", numero, ". Adivina cuál es.")
    
    fallos = 0
    intentos = numero
    aGanado = False
    
    while intentos > 0:
        numeroUsuario = input("Introduce un número: ")
        try:
            numeroUsuario = int(numeroUsuario)
        except ValueError:
            print("Por favor, introduzca un número válido.")
            continue
        
        print("Intentos restantes: ", intentos)
        
        
        if(numeroUsuario == numeroAleatorio):
            print("¡Enhorabuena! Has adivinado el número.")
            aGanado = True
            break
        elif(numeroUsuario < numeroAleatorio):
            print("El número es mayor al introducido.")
            intentos -= 1
            fallos += 1
        elif(numeroUsuario > numeroAleatorio):
            print("El número es menor al introducido.")
            intentos -= 1
            fallos += 1
    if(aGanado):
        print("Has ganado!")
        archivo.write("Puntuación: " + str(calc_puntos(numero,fallos)) + "\n")
    else:
        print("Has perdido.")
        archivo.write("Puntuación: 0\n")
    archivo.close()
    menu()


print("Iniciando el juego de adivinar el número, espere un momento mientras carga.")
sleep(2)
menu()
