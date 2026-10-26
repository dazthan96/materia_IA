import os 
import cv2
import numpy as np
from sklearn.neural_network import MLPClassifier
ruta_0= "E:/facu/07semestre/materia_IA/02_parte/clase02/data/digitos/0/"
archivos = os.listdir(ruta_0)
datos_0 = []
#print(archivos)
for i in archivos:
    archivo = ruta_0 + i
    imagen = cv2.imread(archivo)
    imagen = imagen[:,:,0]
    imagen = np.reshape(imagen,(imagen.shape[0]*imagen.shape[1],1))
    imagen = imagen.T
    datos_0.append(imagen)
datos_0 = np.asarray(datos_0)
print(datos_0.shape)

y_0 = []
y_0 = np.zeros((datos_0.shape[0]))
for i in range(len(y_0)//2):
    y_0[i] = 0
#<---------------------------------------------------------------------------->

ruta_1 = "E:/facu/07semestre/materia_IA/02_parte/clase02/data/digitos/1/"
archivos = os.listdir(ruta_1)
datos_1 = []
#print(archivos)
for i in archivos:
    archivo = ruta_1 + i
    imagen = cv2.imread(archivo)
    imagen = imagen[:,:,0]
    imagen = np.reshape(imagen,(imagen.shape[0]*imagen.shape[1],1))
    imagen = imagen.T
    datos_1.append(imagen)
datos_1 = np.asarray(datos_1)
print(datos_1.shape)
y_1 = []
y_1 = np.zeros((datos_1.shape[0]))
for i in range(len(y_1)//2):
    y_1[i] = 1

red = MLPClassifier(hidden_layer_sizes=(40, 20, 10,1,))
#red.fit(datos_0, y_0)
red.fit(datos_1, y_1)
y_calc = datos_1[4000:5000]
y_calc = red.predict(y_calc)
print(y_calc)
#datos_0[1,:,:]
#datos_0.squeeze(axis=1)
#print(datos_0[1,:])