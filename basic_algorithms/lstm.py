import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error



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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# Reformular os dados para sequência para LSTM (amostras, timesteps, features)
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# Construção do modelo LSTM
model = Sequential()

# Adicionar uma camada LSTM
model.add(LSTM(units=50, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])))

# Camada densa de saída
model.add(Dense(units=1))

# Compilar o modelo
model.compile(optimizer='adam', loss='mean_squared_error')

# Treinar o modelo e armazenar o histórico
history = model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

# Plotar o gráfico de perda (loss)
plt.plot(history.history['loss'])
plt.title('Training Loss')
plt.xlabel('Épocas')
plt.ylabel('Loss')
plt.savefig("lstm.pdf")

# Fazer previsões
y_pred = model.predict(X_test)

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

# Exibir os resultados
print(f"\nMean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Absolute Percentage Error (MAPE): {mape}")
