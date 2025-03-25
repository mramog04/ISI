import urllib.request

url = "https://es.wikipedia.org/wiki/Partido_Socialista_Obrero_Espa%C3%B1ol"


try:
    web = urllib.request.urlopen(url)
except urllib.error.URLError:
    print("La URL no es válida")
    exit()


archivo = open("archivo.txt","w")


for linea in web:
    archivo.write(str(linea.decode("utf-8")))
    
archivo.close()
web.close()

archivo = open("archivo.txt","r")


palabras = 0
en_palabra = False

while True:
    caracter = archivo.read(1)
    if not caracter:
        break
    elif caracter.isspace():
        if en_palabra:
            palabras += 1
            en_palabra = False
    else:
        en_palabra = True
    
print("El número de palabras en el archivo es: ", palabras)