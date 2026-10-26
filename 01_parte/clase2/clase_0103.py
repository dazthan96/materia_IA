import numpy as np
camino = [
    [0, 25, 30, 35, 105],
    [25, 0, 10, 25, 125],
    [30, 10, 0, 20, 120],
    [35, 25, 20, 0, 90],
    [105, 125, 120, 90,0]
]

tablero = np.zeros((5, 5))
suma = 0
individual = [2,0,3,1,4]
for i in range(len(individual)-1):
    print(individual[i], individual[i+1])
    #print(camino[individual[i]][individual[i+1]])
    suma = suma + camino[individual[i]][individual[i+1]]
    
    
print(individual[len(individual)-1], individual[0])
print(suma)
# for number in individual:
#     tablero[individual.index(number)][number] = 3

# print(tablero)        
