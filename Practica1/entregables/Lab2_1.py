temperaturaCelsius = input("Introduzca una temperatura en grados Celsius: ")
while True:
    try:
        temperaturaCelsius = float(temperaturaCelsius)
        break
    except ValueError:
        print("Por favor, introduzca un número válido.")
        temperaturaCelsius = input("Introduzca una temperatura en grados Celsius: ")
temperaturaFahrenheit = (temperaturaCelsius * 9/5) + 32
print("The temperature of %.1f degrees Celsius corresponds to  %.1f degrees Fahrenheit." % (temperaturaCelsius, temperaturaFahrenheit))
