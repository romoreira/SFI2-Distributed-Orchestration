import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Ler o arquivo CSV (substitua 'dados.csv' pelo nome real do seu arquivo)
df = pd.read_csv("dados.csv")

# Exibir as primeiras linhas do DataFrame para verificar se os dados foram lidos corretamente
print(df.head())

# Criando o Interval Plot para a métrica MSE
plt.figure(figsize=(8, 5))
sns.pointplot(x="Algorithm", y="MSE", data=df, ci="sd", capsize=0.2, markers="o", errwidth=1)

# Personalizando o gráfico
plt.xlabel("Algoritmo")
plt.ylabel("MSE")
plt.title("Interval Plot de MSE por Algoritmo")
plt.grid(True)
plt.show()
