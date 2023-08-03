import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn')

# Leia o arquivo CSV
nome_arquivo = '/home/rodrigo/PycharmProjects/SFI2-Distributed-Orchestration/load_gen/reports_to_merge/write_sinusuidal/arquivo_final.csv'  # Substitua pelo caminho do seu arquivo
df = pd.read_csv(nome_arquivo, parse_dates=['time'])

# Defina 'time' como o índice do DataFrame (caso ainda não seja)
df.set_index('time', inplace=True)

# Plote a série temporal
plt.figure(figsize=(10, 6))  # Tamanho do gráfico (opcional)
plt.plot(df.index, df['mean'], label='Mean')  # Plot da série 'mean'
plt.xlabel('Time')  # Rótulo do eixo X
plt.ylabel('Mean')  # Rótulo do eixo Y
plt.title('Write Latency')  # Título do gráfico
plt.legend()  # Mostrar a legenda (opcional)
#plt.grid(True)  # Mostrar as linhas de grade (opcional)
#plt.show()  # Mostrar o gráfico
plt.savefig('mean.pdf')  # Salvar o gráfico como uma imagem (opcional)