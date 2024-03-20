import numpy as np
import sklearn.datasets
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

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

# training the model, printing confusion matrix and accuracy
epoki = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
accuracy_array = []
for e in epoki:
    neuron = Perceptron(max_iter=e, early_stopping=False)
    neuron.fit(X_train, Y_train)
    neuron.predict(X_train)
    Y_predicted = neuron.predict(X_test)
    confusion_matrix_model = confusion_matrix(Y_test, Y_predicted)
    # print(confusion_matrix_model)
    accuracy = (confusion_matrix_model[0, 0] + confusion_matrix_model[1, 1] + confusion_matrix_model[2, 2]) / np.sum(
        confusion_matrix_model)
    print(accuracy)
    accuracy_array.append(accuracy)

plt.plot(epoki, accuracy_array, marker='o')
plt.title('Wpływ liczby epok na dokładność klasyfikacji')
plt.xlabel('Liczba epok')
plt.ylabel('Dokładność klasyfikacji')
plt.show()
