import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

'''Proszę pobrać plik medicine.txt, zawierający wyniki analizy nowego leku. W dwóch pierwszych kolumnach znajduje się
stężenie dwóch składników w próbce krwi, w trzeciej - informacja o tym, czy lek zadziałał. Dane nie są
znormalizowane. Proszę znormalizować dane, podzielić je na zbiór uczący i testujący w proporcjach 80-20 (należy
pamiętać o proporcjach klas), zaproponować wielowarstwową sieć neuronową i zbadać jej skuteczność dla różnych ilości
warstw i neuronów w tych warstwach. Proszę narysować w jaki sposób sieć dokonała podziału w zbiorze dla kilku sieci (
zarówno tych z dobrymi, jak i złymi wynikami) oraz jak wygląda poprawny podział zbioru. Proszę również przedstawić
wyniki dla 5-8 różnych struktur sieci, wraz z oceną, która z nich najlepiej poradziła sobie z zadaniem klasyfikacji.'''

data = pd.read_csv("medicine.txt")  # default data separation by ',' a coma

# Data normalization
scaler = MinMaxScaler()
data = scaler.fit_transform(data)
# print(data)

X = data[:, 0:2]
Y = data[:, 2]

# division into train and test sets with proportion of 80-20
# stratify - ensures proportion of target classes
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, stratify=Y, test_size=0.2)

'''
Jedna warstwa ukryta:
hidden_layer_sizes=(5,)
hidden_layer_sizes=(10,)
hidden_layer_sizes=(20,)

Dwie warstwy ukryte:
hidden_layer_sizes=(5, 5)
hidden_layer_sizes=(10, 10)
hidden_layer_sizes=(20, 20)

Trzy warstwy ukryte:
hidden_layer_sizes=(10, 10, 10)
hidden_layer_sizes=(20, 20, 10)
hidden_layer_sizes=(30, 30, 20)
'''
a = 10
b = 50
c = 100
model = MLPClassifier(hidden_layer_sizes=(a, b, c), max_iter=2000)
model.fit(X_train, Y_train)
Y_predicted = model.predict(X_train)
matrix = confusion_matrix(Y_train, Y_predicted)
print("Results for the train set:")
print(matrix)
print("Accuracy = ", model.score(X_train, Y_train))

Y_test_predicted = model.predict(X_test)
matrix_test = confusion_matrix(Y_test, Y_test_predicted)
print("\nResults for the test set:")
print(matrix_test)
print("Accuracy = ", model.score(X_test, Y_test))

xx, yy = np.meshgrid(np.arange(-0.02, 1.1, 0.01), np.arange(-0.02, 1.1, 0.01))
test_points = np.c_[xx.ravel(), yy.ravel()]

prediction = model.predict(test_points)
prediction = prediction.reshape(xx.shape)
# plt.scatter(test_points[:, 0], test_points[:, 1], c=prediction)
# plt.show()

plt.contourf(xx, yy, prediction, alpha=0.5, cmap=plt.cm.coolwarm)
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.coolwarm, s=20, edgecolors='k', linewidths=0.9)
plt.xlabel('Presence 1')
plt.ylabel('Presence 2')
plt.title('Ilosc warstw i sieci ({}, {}, {})'.format(a, b, c))
plt.show()
