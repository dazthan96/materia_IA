import cv2
import numpy as np
archivo = "C:/Users/Beto/Documents/avance 2026-II/IA/02_parte/data/digitos/1/img_0.jpg"
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