from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


"""
Zadanie 5.

Proszę sprawdzić wyniki regresji dla zbioru california housing dla różnych podziałów na dane uczące i testujące (dla
co najmniej pięciu podziałów 20-80, 35-65, 50-50, 65-35, 80-20) i wyciągnąć wnioski.
"""


X, Y = fetch_california_housing(return_X_y=True)
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

test_sizes = []
n = 0.05
for i in range(19):
    test_sizes.append(n)
    n += 0.05
mse = []
score = []

for test_size in test_sizes:
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size)
    network = MLPRegressor(hidden_layer_sizes=(80, 80, 80, 80), max_iter=20000, activation='relu', solver='adam')
    network.fit(X_train, Y_train)
    y_pred = network.predict(X_test)
    mse.append(mean_squared_error(Y_test, y_pred))
    score.append(network.score(X_test, Y_test))

plt.scatter(test_sizes, mse, label='mse', color='red')
plt.scatter(test_sizes, score, label='score', color='blue')
plt.legend()
plt.xlabel("test set size")
plt.ylabel("mse & score")

plt.show()

