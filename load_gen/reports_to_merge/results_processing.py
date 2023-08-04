import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_dataset():
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

def plot_coef():
    nome_arquivo = '/home/rodrigo/PycharmProjects/SFI2-Distributed-Orchestration/load_gen/reports_to_merge/write_sinusuidal/arquivo_final.csv'  # Substitua pelo caminho do seu arquivo
    df = pd.read_csv(nome_arquivo)

    # Visualizar as primeiras linhas do DataFrame para entender os dados
    print(df.head())

    # Calcular a matriz de correlação
    correlation_matrix = df.corr()

    # Visualizar a matriz de correlação em um heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Matriz de Correlação")
    plt.show()
    # Calcular a matriz de correlação
    correlation_matrix = df.corr()

    # Exibir a tabela com os valores numéricos das correlações
    print(correlation_matrix)
    # Criar um ranking das colunas com base na correlação com uma coluna de interesse (por exemplo, a primeira coluna do DataFrame)
    target_column = df.columns['mean']  # Substitua '0' pelo índice da coluna que você deseja correlacionar com as outras
    correlation_ranking = correlation_matrix[target_column].abs().sort_values(ascending=False)

    print(correlation_ranking)

plot_coef()