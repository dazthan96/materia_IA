import os 
import cv2
import numpy as np
ruta = "C:/Users/Beto/Documents/avance 2026-II/IA/02_parte/data/digitos/1/"
archivos = os.listdir(ruta)
datos = []
print(archivos)
for i in archivos:
    archivo = ruta + i
    imagen = cv2.imread(archivo)
    imagen = imagen[:,:,0]
    imagen = np.reshape(imagen,(imagen.shape[0]*imagen.shape[1],1))
    imagen = imagen.T
    datos.append(imagen)
datos = np.asarray(datos)
print(datos.shape)
#cv2.imshow('otra imagen',canal1)
#cv2.waitKey(0)
#cv2.destroyAllWindows()