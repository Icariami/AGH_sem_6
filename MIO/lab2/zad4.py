import time

from numpy import ravel
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# fetch dataset
yeast = fetch_ucirepo(id=110)

# data (as pandas dataframes)
X = yeast.data.features
Y = yeast.data.targets

# changing labels from strings to numbers
le = LabelEncoder()
Y = ravel(Y)
Y = le.fit_transform(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y)


'''
hidden layer sizes
activation function - default is relu, also possible values are: {'identity', 'tanh', 'logistic', 'relu'}
solver - default is 'adam', also possible values are 'sgd', 'lbfgs'
learning_rate - default is 'constant', also possible values are 'adaptive', 'invscaling'
relu + adam + adaptive ~ 60%
'''
model = MLPClassifier(hidden_layer_sizes=(10, 20, 50), max_iter=20000,
activation='tanh',
solver='lbfgs',
learning_rate='invscaling'
)
start = time.time()
model.fit(X_train, Y_train)
end = time.time()
print('Training time:', end - start, 'seconds')

print("Results for the train set:")
Y_pred_train = model.predict(X_train)
matrix2 = confusion_matrix(Y_train, Y_pred_train)
print(matrix2)
print("Accuracy = ", model.score(X_train, Y_train))

print("\nResults for the test set:")
Y_pred = model.predict(X_test)
matrix = confusion_matrix(Y_test, Y_pred)
print(matrix)
print("Accuracy = ", model.score(X_test, Y_test))



