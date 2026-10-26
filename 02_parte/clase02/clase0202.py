import os 
import cv2
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
ruta = "E:/facu/07semestre/materia_IA/02_parte/clase02/data/digitos/1"
archivos = os.listdir(ruta)
datos = []
#print(archivos)
for i in archivos:
    archivo = ruta + i
    imagen = cv2.imread(archivo)
    imagen = imagen[:,:,0]
    imagen = np.reshape(imagen,(imagen.shape[0]*imagen.shape[1],1))
    imagen = imagen.T
    imagen_plana = imagen.flatten()
    datos.append(imagen_plana)
datos = np.asarray(datos)
print(datos.shape)
#datos[1,:,:]
#datos.squeeze(axis=1)

y = []
y = np.ones((datos.shape[0]))
red = MLPClassifier(hidden_layer_sizes=(40,20,10,))
red.fit(datos, y)
y_calc = datos[4000:5000]
y_calc = red.predict(y_calc)
y_real = y[4000:5000]
print(accuracy_score(y_true=y_real, y_pred=y_calc))
#cv2.imshow('otra imagen',canal1)
#cv2.waitKey(0)
#cv2.destroyAllWindows()