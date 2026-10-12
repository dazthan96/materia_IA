#   a   b   c   d   e 
#a 000 025 030 035 105
#b 025 000 010 025 125
#c 030 010 000 020 120
#d 035 025 020 000 090
#e 105 125 120 090 000
import numpy as np
import random
#Algoritmos del agente viajero

matriz = np.array([[0, 25, 30, 35, 105],[25, 0, 10, 25, 125],[30, 10, 0, 20, 120],[35, 25, 20, 0, 90],[105, 125, 120, 90,0]])
print(matriz)
lista = [1,2,3,4,5]
import numpy as np

# 1. Crear una cuadrícula de coordenadas (asigna indexing='ij' para mantener el orden matricial)
grid = np.meshgrid(lista, lista, indexing='ij')
# 2. Convertir las matrices resultantes en columnas y apilarlas
combinaciones = np.stack(grid, axis=-1).reshape(-1, 2)

print(combinaciones)
