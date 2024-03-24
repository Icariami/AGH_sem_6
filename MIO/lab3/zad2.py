import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

"""
Proszę zaproponować jak najmniejszą sieć (najlepiej z jedną warstwą ukrytą) do aproksymacji funkcji  f(x)=sin(x)  w
przedziale  [−2π,2π] . Proszę użyć  tanh  jako funkcji aktywacji. Proszę narysować funkcję aproksymowaną i
aproksymującą. Wykorzystując dostęp do wag i biasów (network.coefs_ i network.intercepts_) proszę zapisać wzór
funkcji aproksymującej.
"""

X = np.arange(-2 * np.pi, 2 * np.pi, 0.01)
Y = np.sin(X)

X_train, X_test, Y_train, Y_test = train_test_split(X.reshape(-1, 1), Y, test_size=0.2)
network = MLPRegressor(hidden_layer_sizes=(30), max_iter=20000, activation='tanh')
network.fit(X_train, Y_train)
y_pred = network.predict(X.reshape(-1, 1))

print("train score: ", network.score(X_train, Y_train))
print("test score: ", network.score(X_test, Y_test))

plt.plot(X, Y, 'b', label="f. aproksymowana")
plt.plot(X, y_pred, 'r', label="f. aproksymująca")
plt.legend(loc='upper right')
plt.show()

print("y = ")
for index in range(30):
    print(f"{network.coefs_[0][0][index]:.2f}*tanh({network.coefs_[1][index][0]:.2f}x + {network.intercepts_[0][index]:.2f}) + ")
