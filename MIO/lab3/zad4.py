from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

"""
Zadanie 4.

Proszę, bazując na powyższym zbiorze danych, dla wybranych struktur sieci (np. najlepszej otrzymanej sieci),
wykonać wykresy zależności ilości wykonanych przez sieć epok oraz uzyskanych metryk. Uzyskany wynik należy
odpowiednio opisać oraz odnieść do dotychczasowych zagadnień poruszanych na zajęciach.
"""

X, Y = fetch_california_housing(return_X_y=True)
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
network = MLPRegressor(hidden_layer_sizes=(80, 80, 80, 80), max_iter=20000, activation='relu', solver='adam')

epochs = []
mse = []
score = []

for epoch in range(1, 200):
    network.partial_fit(X_train, Y_train)
    y_pred = network.predict(X_test)

    epochs.append(epoch)
    mse.append(mean_squared_error(Y_test, y_pred))
    score.append(network.score(X_test, Y_test))

plt.scatter(epochs, mse, label='mse', color='red', s=20, edgecolors='black', linewidths=0.5)
plt.scatter(epochs, score, label='score', color='blue', s=20, edgecolors='black', linewidths=0.5)
plt.xlabel("number of epochs")
plt.ylabel("mse & score")
plt.legend()
plt.show()
