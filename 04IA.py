from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
data = load_iris()
X = data.data
y = data.target

X_train, x_test, y_train, y_test = train_test_split(X,y,train_size=0.8)
#print(X_train.shape, x_test.shape)
#print(y_train)
#<--definicion del modelo claificador-->
model = DecisionTreeClassifier()
#<--datos de entrenamiento-->
model.fit(X_train, y_train)
y_pred = model.predict(x_test)
print(y_pred)
print(y_test)