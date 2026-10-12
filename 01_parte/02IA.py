from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, Normalizer, LabelEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
import numpy as np

matriz_original = np.array([[1,2,3,4],[1,np.nan,2,4],[np.nan, 3,8,7]])
print('<--------------matriz original-------------->')
print(matriz_original)

prepro1 = SimpleImputer(strategy='mean')
matriz_imputada = prepro1.fit_transform(matriz_original)
print('<--------------matriz imputada-------------->')
print(matriz_imputada)

prepro2 = StandardScaler()
matriz_standarizada = prepro2.fit_transform(matriz_imputada)
print('<--------------stardarscaler-------------->')
print(matriz_standarizada)

prepro3 = Normalizer()
matriz_normalizada = prepro3.fit_transform(matriz_standarizada)
print('<--------------normalizer-------------->')
print(matriz_normalizada)
#prepo 2 y 3 es igual a la matriz de entrada
#listaprepro = [prepro1, prepro2,prepro3]
matriz = matriz_original
#for i in listaprepro:
#    matriz = i.fit_transform(matriz)
#    print(matriz)

tuberia = Pipeline([
    ('imputador', SimpleImputer(strategy='mean')),
    ('estandarizador', StandardScaler()),
    ('normalizador', Normalizer())
])

# 2. Ejecutamos todo el flujo de un solo golpe
matriz_final = tuberia.fit_transform(matriz)

print(matriz_final)

#buscar e investigar pipeline sklearn
lista = ['strongay','podereoso bolivar','wilsterman comelon']
prepro4 = LabelEncoder()
lista2 = prepro4.fit_transform(lista)
print('label encoder')
print(lista2)

prepro5 = OneHotEncoder()
lista3 = prepro5.fit_transform(lista2.reshape(-1,1)).toarray()
print('OneHotEncoder')
print(lista3)