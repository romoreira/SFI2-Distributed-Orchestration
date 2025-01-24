import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb  # Importando o XGBoost

import os
import argparse

import csv


# Configura o argparse para receber o argumento --file-name
parser = argparse.ArgumentParser(description="Ler um arquivo CSV especificado pelo usuário.")
parser.add_argument("--file-name", type=str, required=True, help="Nome do arquivo CSV a ser lido")
args = parser.parse_args()
# Caminho completo do arquivo
csv_path = os.path.join(os.getcwd(), args.file_name)
df = pd.read_csv(csv_path)

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

# Criar e treinar o modelo XGBoost
xgboost_model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, learning_rate=0.1)
xgboost_model.fit(X_train, y_train.ravel())

# Fazer previsões
y_pred = xgboost_model.predict(X_test)
y_pred = y_pred.reshape(-1, 1)  # Transforma y_pred em uma matriz 2D

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
print("=== Results for XGBoost ===")
print(f"\nMean Squared Error (MSE): {mse:.10f}")
print(f"Mean Absolute Error (MAE): {mae:.10f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.10f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.10f}")



# Nome do arquivo CSV
csv_file = str(args.file_name).split('.')[0]+"_resultados_basic_algorithms-XGBoost-.csv"

csv_file = os.path.join("results", csv_file)
# Verificar se o arquivo já existe, para adicionar o cabeçalho apenas na primeira vez
header = ["MSE", "MAE", "RMSE", "MAPE"]

try:
    # Tentar abrir o arquivo no modo append
    with open(csv_file, mode="a", newline="") as file:
        writer = csv.writer(file)

        # Se o arquivo está vazio, adicionar o cabeçalho
        if file.tell() == 0:
            writer.writerow(header)

        # Formatar os resultados para evitar notação científica
        formatted_results = [f"{value:.10f}" if isinstance(value, float) else value for value in [mse, mae, rmse, mape]]
        
        # Adicionar os resultados no arquivo
        writer.writerow(formatted_results)

    print(f"Resultados salvos no arquivo '{csv_file}'.")
except Exception as e:
    print(f"Erro ao salvar resultados no arquivo: {e}")