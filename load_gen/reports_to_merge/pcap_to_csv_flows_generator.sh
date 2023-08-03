#!/bin/bash

# Defina o caminho do diretório onde estão os arquivos .pcap
diretorio="./write_sinusuidal/pcaps"

# Percorre o diretório e executa o script Python para cada arquivo .pcap encontrado
for arquivo in "$diretorio"/*.pcap; do
    # Obtém o nome do arquivo sem a extensão para uso no nome do arquivo de saída .csv
    nome_arquivo=$(basename "${arquivo%.*}")
    echo $arquivo
    python3 /home/rodrigo/PycharmProjects/FlowGenerator/main.py -f "$arquivo" --csv --output-file "write_sinusuidal/csvs/cassandra_flows/${nome_arquivo}.csv"
done
    # Executa o script Python passando os parâmetros necessários
