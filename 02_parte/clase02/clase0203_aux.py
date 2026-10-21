import os 
import cv2
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
ruta_1 = "C:/Users/Beto/Documents/avance 2026-II/IA/02_parte/data/digitos/1/"
archivos_1 = os.listdir(ruta_1)
x_train = []
y_train = []
for i in archivos_1:
    archivo = ruta_1 + i
    imagen = cv2.imread(archivo)
    imagen = imagen[:,:,0]
    imagen_plana = imagen.flatten()
    x_train.append(imagen_plana)
    y_train.append(1)
ruta_0 = "C:/Users/Beto/Documents/avance 2026-II/IA/02_parte/data/digitos/0/"
archivos_0 = os.listdir(ruta_0)
for i in archivos_0:
    archivo = ruta_0 + i
    imagen = cv2.imread(archivo)
    imagen = imagen[:,:,0]
    imagen_plana = imagen.flatten()
    x_train.append(imagen_plana)
    y_train.append(0)
x_train_completo = np.asarray(x_train)
y_train_completo = np.asarray(y_train)

x_train, x_test, y_train, y_test =train_test_split(
    x_train_completo,
    y_train_completo,
    test_size=0.20,
    random_state=42
)
red = MLPClassifier(hidden_layer_sizes=(40,20,10,) )
red.fit(x_train, y_train)


y_calc = red.predict(x_test)
print(accuracy_score(y_true=y_test, y_pred=y_calc))