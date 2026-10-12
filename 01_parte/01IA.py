import numpy as np

lista1= [i for i in range(10,2,-1)]
print(lista1)
def suma(a:int,b:int):
    return a+b

def multi(a:int,b:int):
    return np.sum([a for u in range(b)])

print(multi(4,3))

#hacer la normalizacion y discretizacion matematicamente con todos los paramatros(tambien programarsin libreria)