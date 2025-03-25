import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Solicitar el número de partículas
num_particulas = int(input("Introduce el número de partículas del sistema: "))

# Diccionario para almacenar los datos de las partículas
particles = {}

for i in range(num_particulas):
    x = float(input(f"Introduce la posición x de la partícula {i+1}: "))
    y = float(input(f"Introduce la posición y de la partícula {i+1}: "))
    mass = float(input(f"Introduce la masa de la partícula {i+1}: "))
    particles[f'Particle_{i+1}'] = {'x': x, 'y': y, 'mass': mass}

# Convertir el diccionario en un DataFrame de pandas
df = pd.DataFrame.from_dict(particles, orient='index')

# Calcular el centro de masas
cx = np.sum(df['x'] * df['mass']) / np.sum(df['mass'])
cy = np.sum(df['y'] * df['mass']) / np.sum(df['mass'])

# Mostrar el resultado
print(f"El centro de masas está en: ({cx}, {cy})")

plt.scatter(df['x'], df['y'], s=80, c='blue', alpha=0.6, label="Partículas")
for i, (x, y, mass) in enumerate(zip(df['x'], df['y'], df['mass']), 1):
    plt.text(x, y, f"M={mass}", fontsize=9, ha='right', color='black')

# Graficar el centro de masas como un triángulo rojo
plt.scatter(cx, cy, marker='^', color='green', s=80, label="Centro de masas")
plt.text(cx, cy, f"M={np.sum(df['mass'])/df['mass'].shape}", fontsize=12, ha='right', color='black')

# Etiquetas y leyenda
plt.xlabel("Posición X")
plt.ylabel("Posición Y")
plt.legend()
plt.grid(True)

# Mostrar la gráfica
plt.show()
