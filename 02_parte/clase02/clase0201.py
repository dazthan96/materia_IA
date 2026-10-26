import cv2
import numpy as np
archivo = "E:/facu/07semestre/materia_IA/02_parte/clase02/data/digitos/0/img_1.jpg"
imagen = cv2.imread(archivo)
print(imagen.shape)
canal1 = imagen[:,:,0]
cv2.imshow('imagen original', imagen)
cv2.imshow('Canal 1', canal1)
canal1 = np.reshape(canal1, (canal1.shape[0]*canal1.shape[1],1))
canal1 = canal1.T
cv2.imshow('otra imagen',canal1)
cv2.waitKey(0)

cv2.destroyAllWindows()