import sys

def convertirMayusculas(frase):
    frase = frase.upper()
    return frase

def convertirMinusculas(frase):
    frase = frase.lower()
    return frase

def covertirPrimeraLetraMayuscula(frase):
    lista = frase.rsplit(" ")
    for i in range(len(lista)):
        lista[i] = lista[i].capitalize()
    lista = " ".join(lista)
    frase = lista
    return lista

def convertirPosicionesParesMayusculas(frase):
    lista = frase.rsplit(" ")
    for i in range(len(lista)):
        aux = list(lista[i])
        for j in range(len(aux)):
            if(j%2==0):
                if(aux[j].islower()):
                    aux[j] = aux[j].upper()
        aux = "".join(aux)
        lista[i] = aux
    lista = " ".join(lista)
    frase = lista
    return lista

def salir(frase):
    print("Cerrando el programa")
    sys.exit()

                
frase = input("Introduzca una frase: ")
while True:
    print("Menu:")
    print("1. Convert the sentence into uppercase.")
    print("2. Convert the sentence into lowercase")
    print("3. Convert the first character of each word into uppercase.")
    print("4. Convert the characters that are in even positions into uppercase.")
    print("5. Exit")
    inputMenu = input("Introduzca una opcion: ")
    
    opciones = {
        "1": convertirMayusculas,
        "2": convertirMinusculas,
        "3": covertirPrimeraLetraMayuscula,
        "4": convertirPosicionesParesMayusculas,
        "5": salir
    }
    
    if inputMenu in opciones:
        print(opciones[inputMenu](frase))
    else:
        print("Opcion no valida")
    
