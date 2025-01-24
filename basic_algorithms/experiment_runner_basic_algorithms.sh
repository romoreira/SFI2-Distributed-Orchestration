#!/bin/bash

# Lista dos scripts Python
scripts=("cat_boost.py" "dt.py" "knn.py" "lstm.py" "rf.py" "xboost_.py")

# Lista dos arquivos CSV
csv_files=("read_fabric.csv" "write_fibre.csv" "write_fabric.csv" "read_fibre.csv")

# Número de execuções por script e arquivo CSV
repeat=10

# Loop para cada script
for script in "${scripts[@]}"; do
  # Loop para cada arquivo CSV
  for csv_file in "${csv_files[@]}"; do
    # Executa o script Python 10 vezes com o CSV atual
    for ((i=1; i<=repeat; i++)); do
      echo "Executando: python3 $script --file-name $csv_file (Execução $i)"
      python3 "$script" --file-name "$csv_file"
    done
  done
done
