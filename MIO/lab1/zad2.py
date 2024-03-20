import numpy as np
import sklearn.datasets
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import random

'''
Proszę pobrać zbiór https://archive.ics.uci.edu/ml/datasets/iris.
Można to też zrobić w pythonie używając funkcji sklearn.datasets.load_iris( ).
Następnie proszę dokonać samodzielnego podziału na dane uczące i testujące w
proporcji 80%/20%. Proszę zbudować sieć złożoną z pojedynczej warstwy perceptronów
(np. używając omawianej już tutaj funkcji sklearn.linear_model.Perceptron),
której zadaniem będzie jak najdokładniejsza klasyfikacja gatunków irysów na
podstawie ich pomiarów. Proszę dokonać analizy macierzy pomyłek dla kilku
uruchomień algorytmu. Jaką największą trafność jest w stanie uzyskać pojedyncza
warstwa perceptronów w tym zadaniu? Dlaczego? (Podpowiedź: polecamy przyjrzeć się
pojęciu liniowej separowalności).
'''

# data_set is a 'bunch', sth like a dictionary
data_set = sklearn.datasets.load_iris()

# shape: matrix (150, 4), 150 irises in 3 different classes
X = data_set.data

# classification data
Y = data_set.target

# column's names (4 different columns)
feature_names = data_set.feature_names
print(feature_names)

# names of target classes (3 classes)
target_names = data_set.target_names
print(target_names)

# first way of dividing train and test sets
# random selection from the whole set (can be different number of elements per class)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=30)

train_test_sets = [(X_train, Y_train, X_test, Y_test)]

# second way of dividing train and test sets
X_test_2 = []
Y_test_2 = []
X_train_2 = []
Y_train_2 = []

r = random.randint(0, 40)
for i in range(150):
    if r <= i < r+10 or r+50 <= i < r+60 or r+100 <= i < r+110:
        X_test_2.append(X[i])
        Y_test_2.append(Y[i])
    else:
        X_train_2.append(X[i])
        Y_train_2.append(Y[i])

train_test_sets.append((X_train_2, Y_train_2, X_test_2, Y_test_2))

# third way of dividing train and test sets
random_nums_0 = random.sample(range(0, 49), 10)
random_nums_1 = random.sample(range(50, 99), 10)
random_nums_2 = random.sample(range(100, 149), 10)

X_test_3 = []
Y_test_3 = []
X_train_3 = []
Y_train_3 = []

for i in range(150):
    if i in random_nums_0 or i in random_nums_1 or i in random_nums_2:
        X_test_3.append(X[i])
        Y_test_3.append(Y[i])
    else:
        X_train_3.append(X[i])
        Y_train_3.append(Y[i])

train_test_sets.append((X_train_3, Y_train_3, X_test_3, Y_test_3))


# training the model, printing confusion matrix and accuracy
for (X_train, Y_train, X_test, Y_test) in train_test_sets:
    accuracy = 0
    for i in range(100):
        neuron = Perceptron(early_stopping=False)
        neuron.fit(X_train, Y_train)
        neuron.predict(X_train)
        Y_predicted = neuron.predict(X_test)
        confusion_matrix_model = confusion_matrix(Y_test, Y_predicted)
        # print(confusion_matrix_model)
        accuracy += (confusion_matrix_model[0, 0] + confusion_matrix_model[1, 1] + confusion_matrix_model[2, 2]) / np.sum(confusion_matrix_model)

    print(accuracy/100)
