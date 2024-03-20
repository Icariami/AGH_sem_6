from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

data_set = datasets.load_digits()
X = data_set.data
Y = data_set.target

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y)

'''
hidden layer sizes
activation function - default is relu, also possible values are: {'identity', 'tanh', 'logistic', 'relu'}
solver - default is 'adam', also possible values are 'sgd', 'lbfgs'
learning_rate - default is 'constant', also possible values are 'adaptive', 'invscaling'
'''
model = MLPClassifier(hidden_layer_sizes=(10, 20, 50), max_iter=20000, activation='logistic', solver='sgd', learning_rate='adaptive')

model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
matrix = confusion_matrix(Y_test, Y_pred)
print(matrix)
print("Accuracy = ", model.score(X_test, Y_test))
#
# Y_predicted = model.predict(X_train)
# matrix2 = confusion_matrix(Y_train, Y_predicted)
# print(matrix2)
# print(model.score(X_train, Y_train))

