from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

data = load_iris()
datos_entrada= data.data
datos_salida= data.target

x_train, x_test, y_train, y_test = train_test_split(datos_entrada,datos_salida, train_size=0.8)
clasificador = MLPClassifier(hidden_layer_sizes=(10,100,))
clasificador.fit(X=x_train, y=y_train)
print(clasificador.predict(x_test))
print(y_test)