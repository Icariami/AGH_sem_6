import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import MinMaxScaler

"""

Proszę pobrać zbiór california housing (można go załadować w sklearnie dzięki funkcji fetch_california_housing( ),
jest też dostępny w colabie jako przykładowy plik testowy w colabie). Zawiera nieznormalizowane dane dotyczące bloków
mieszkalnych w Kaliforni zebranych w 1990 roku, takie jak lokacja geograficzna, wiek, całkowita liczba mieszkańców
bloku, znajdujące się w nim mieszkania i sypialnie, oraz mediana dochodów tych mieszkańców. W ostatniej kolumnie
znajduje się mediana wartości mieszkania w tym bloku. Proszę znormalizować dane, a następnie zaproponować kilka
wielowarstowych sieci neuronowych i ocenić jak dobrze dokonają aproksymacji funkcji mediany wartości mieszkań za
pomocą opisanych dzisiaj metryk. Proszę spróbować osiągnąć jak najlepszy wynik (jak najniższe MSE). Wyniki oczywiście
proszę sprawdzać na danych testujących."""

X, Y = fetch_california_housing(return_X_y=True)
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

hidden_layers = [(20, 20), (50, 50, 50), (80, 80, 80, 80), (100, 100, 100, 100, 100)]
activations = ['relu', 'tanh']

for hidden_layer in hidden_layers:
    for activation in activations:
        network = MLPRegressor(hidden_layer_sizes=hidden_layer, activation=activation, max_iter=20000, solver='adam')
        network.fit(X_train, Y_train)
        y_pred = network.predict(X_test)

        print(f"Results for: hidden_layer_sizes = {hidden_layer}, activation = {activation}")
        print("score: ", round(network.score(X_test, Y_test), 3))
        print("mse: ", round(mean_squared_error(Y_test, y_pred), 3), "\n")



