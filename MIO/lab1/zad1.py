import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import random

'''
Proszę wygenerować po 400 2-wymiarowych punktów przypisanych do dwóch klas
K1 i K2, pochodzących z rozkładów normalnych N([0,-1],1) i N([1,1],1)
i podzielić je losowo na zbiory uczące i testujące w proporcji N do 400-N.
Proszę sprawdzić średnią dokładność klasyfikacji i podać odchylenie
standardowe dla N = 2, 5, 10, 20, 100, powtarzając eksperyment 10 razy dla
każdego N. Dla każdego N dla jednej z powtórek proszę ustalić wzór
hiperpłaszczyzny (w naszym wypadku - prostej) oddzielającej klasy, a
następnie pokazać ją na wykresie razem z danymi (w sumie 5 wykresów).
'''

# 2D points
X = np.concatenate((np.random.normal([0, -1], [1, 1], [400, 2]), np.random.normal([1, 1], [1, 1], [400, 2])))

# K1 and K2 classes
Y = np.concatenate((np.array([[0, 0]] * 400), np.array([[0, 1]] * 400)))

N = [2, 5, 10, 20, 100]

# for every N value
for i in N:
    accuracy_array = []
    printData = True

    # 10 reps for every N value
    for _ in range(10):

        # divide into learning and testing sets
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=i)
        neuron = Perceptron()
        neuron.fit(X_train, Y_train[:, 1])
        neuron.predict(X_train)
        Y_predicted = neuron.predict(X_test)

        # confusion matrix
        confusion_matrix_model = confusion_matrix(Y_test[:, 1], Y_predicted, labels=[True, False])

        accuracy = (confusion_matrix_model[0, 0] + confusion_matrix_model[1, 1]) / np.sum(confusion_matrix_model)
        accuracy_array.append(accuracy)

        # plot once for a N value
        if printData:
            print("a")
            plt.figure()
            plt.scatter(X_test[:, 0], X_test[:, 1], c=Y_test[:, 1], cmap='cividis', edgecolors='k')
            plt.title(f"Hiperpłaszczyzna dla N={i}")
            x1 = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100)

            # hyperplane equation
            x2 = -(1. / neuron.coef_[0][1]) * (neuron.coef_[0][0] * x1 + neuron.intercept_[0])
            plt.plot(x1, x2, color='red')
            plt.show()
            printData = False

    print(f"\nN = {i}")
    print("Mean = " + str(np.mean(accuracy_array)))
    print("Standard deviation " + str(np.std(accuracy_array)))
