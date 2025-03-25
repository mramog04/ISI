def menu():
    print("1. Sumar")
    print("2. Restar")
    print("3. Multiplicar")
    print("4. Dividir")
    print("5. Potencia")
    print("6. Logaritmo neperiano")
    print("7. Salir")
    opcion = input("Introduce una opción: ")
    while True:
        try:
            opcion = int(opcion)
            if opcion < 1 or opcion > 7:
                print("Por favor, introduzca una opción válida.")
            else:
                break
        except ValueError:
            print("Por favor, introduzca una opción válida.")
        opcion = input("Introduce una opción: ")
    if(opcion == 1):
        sumar()
    elif(opcion == 2):
        restar()
    elif(opcion == 3):
        multiplicar()
    elif(opcion == 4):
        dividir()
    elif(opcion == 5):
        potencia()
    elif(opcion == 6):
        logaritmo()
    elif(opcion == 7):
        print("Gracias por usar la calculadora.")
        exit()

def sumar():
    lista = []
    print("Introduce los números que deseas sumar. Introduce 'q' para realizar la operación.")
    numero = input("Introduzca un valor: ")
    while True:
        try:
            if numero == "q":
                break
            else:
                numero = float(numero)
                lista.append(numero)
        except ValueError:
            print("Por favor, introduzca un número válido.")
        numero = input("Introduzca un valor: ")
    suma = sum(lista)
    print("El resultado de la suma es: ", suma)
    menu()

def restar():
    numero1 = input("Introduzca el minuendo: ")
    numero2 = input("Introduzca el sustraendo: ")
    while True:
        try:
            numero1 = float(numero1)
            numero2 = float(numero2)
            break
        except ValueError:
            print("Por favor, introduzca un número válido.")
        numero1 = input("Introduzca el minuendo: ")
        numero2 = input("Introduzca el sustraendo: ")
    resta = numero1 - numero2
    print("El resultado de la resta es: ", resta)
    menu()

def multiplicar():
    lista = []
    print("Introduce los números que deseas multiplicar. Introduce 'q' para realizar la operación.")
    numero = input("Introduzca un valor: ")
    while True:
        try:
            if numero == "q":
                break
            else:
                numero = float(numero)
                lista.append(numero)
        except ValueError:
            print("Por favor, introduzca un número válido.")
        numero = input("Introduzca un valor: ")
    for n in range(len(lista)):
        if n == 0:
            producto = lista[n]
        else:
            producto = producto * lista[n]
    print("El resultado de la multiplicacion es: ", producto)
    menu()
    
def dividir():
    dividendo = input("Introduzca el dividendo: ")
    divisor = input("Introduzca el divisor: ")
    while True:
        try:
            dividendo = float(dividendo)
            divisor = float(divisor)
            break
        except ValueError:
            print("Por favor, introduzca un número válido.")
        dividendo = input("Introduzca el dividendo: ")
        divisor = input("Introduzca el divisor: ")
    if divisor == 0:
        print("No se puede dividir entre 0.")
    else:
        division = dividendo / divisor
        print("El resultado de la division es: ", division)
    menu()

def potencia():
    base = input("Introduzca la base: ")
    exponente = input("Introduzca el exponente: ")
    while True:
        try:
            base = float(base)
            exponente = float(exponente)
            break
        except ValueError:
            print("Por favor, introduzca un número válido.")
        base = input("Introduzca la base: ")
        exponente = input("Introduzca el exponente: ")
    potencia = base ** exponente
    print("El resultado de la potencia es: ", potencia)
    menu()
    
def logaritmo():
    numero = input("Introduzca el número: ")
    while True:
        try:
            numero = float(numero)
            if numero <= 0:
                print("El logaritmo de un número negativo o 0 no es posible.")
            else:
                break
        except ValueError:
            print("Por favor, introduzca un número válido.")
        numero = input("Introduzca el número: ")
    import math
    logaritmo = math.log(numero)
    print("El resultado del logaritmo neperiano es: ", logaritmo)
    menu()
    
def main():
    print("Bienvenido a la calculadora.")
    menu()

if __name__ == "__main__":
    main()