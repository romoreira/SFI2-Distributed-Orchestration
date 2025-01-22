import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os

df = pd.read_csv("write.csv")

# Visualizar as primeiras linhas
print("Dados carregados:")
print(df.head())

df = df.drop(columns=['time'])

df.drop(columns='type total', inplace=True)
X = df.iloc[:, [i for i in range(df.shape[1]) if i != 4]]  # Todas as colunas exceto a quarta
y = df.iloc[:, 4]  # Quarta coluna (índice 3)

# Remover linhas com valores ausentes
print(f"Antes da remoção: {X.shape[0]} linhas")
data = pd.concat([X, y], axis=1)
data = data.dropna()

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

print(f"Depois da remoção: {X.shape[0]} linhas")

# Normalizar as features e o rótulo para o intervalo [0, 1]
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y.values.reshape(-1, 1))  # reshape para garantir que y seja uma coluna

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Criar e treinar o modelo KNN
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)

# Fazer previsões
y_pred = knn.predict(X_test)

# Calcular MSE, MAE, RMSE e MAPE
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)

# Ignorar os casos onde y_test é zero no cálculo do MAPE
non_zero_mask = y_test != 0  # Máscara para selecionar apenas valores diferentes de zero
y_test_non_zero = y_test[non_zero_mask]
y_pred_non_zero = y_pred[non_zero_mask]

# Calcular MAPE apenas para os casos onde y_test não é zero
mape = np.mean(np.abs((y_test_non_zero - y_pred_non_zero) / y_test_non_zero)) * 100

print("==Results for KNN===")
# Exibir os resultados
print(f"\nMean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Absolute Percentage Error (MAPE): {mape}")
