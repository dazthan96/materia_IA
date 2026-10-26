from sklearn.neural_network import MLPClassifier

datos_entrada = [[0,0],[0,1],[1,0],[1,1]]
datos_salida = [0,0,0,1]

clasificador = MLPClassifier(hidden_layer_sizes=(10,10,))
clasificador.fit(X=datos_entrada, y=datos_salida)

print(clasificador.predict([[0,1],[1,1]]))