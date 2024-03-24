import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler

"""
Plik advertising.csv zawiera w każdym rzędzie informację na temat wydatków na reklamę telewizyjną, reklamową i
prasową dla pojedynczego produktu oraz zyski z jego sprzedaży. Można przedstawić zyski jako funkcję  Z(wTV,wradio,
wprasa) . Proszę zaproponować architrekturę sieci neuronowej, która dokona aproksymacji tej funkcji i dokonać
ewaluacji tej sieci. Proszę porównać wyniki (MSE) dla przynajmniej dwóch różnych struktur jeżeli chodzi o liczbę
neuronów i dla dwóch różnych funkcji aktywacji (najlepiej relu i tanh). Proszę pamiętać o podzieleniu zbioru na dane
uczące i testujące.
"""

data = pd.read_csv("Advertising.csv")

X = data[['TV', 'Radio', 'Newspaper']]
Y = data['Sales']

# data normalization - normalization of X only
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

hidden_layers = [(20, 20, 20), (50, 50, 50, 50), (80, 80, 80, 80, 80)]
activations = ['relu', 'tanh']

for hidden_layer in hidden_layers:
    for activation in activations:
        network = MLPRegressor(hidden_layer_sizes=hidden_layer, activation=activation, max_iter=20000)
        network.fit(X_train, y_train)
        y_pred = network.predict(X_test)

        print(f"Results for: hidden_layer_sizes = {hidden_layer}, activation = {activation}")
        print("score: ", network.score(X_test, y_test))
        print("mse: ", mean_squared_error(y_test, y_pred))
