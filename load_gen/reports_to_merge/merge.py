import csv
import os
import pandas as pd
from datetime import datetime, timedelta

def write_to_csv(df, output_file):
    df.to_csv(output_file, index=False)
def extract_csv_pattern(input_directory, phrase_pattern, output_directory):
    amout_files = 0
    for filename in os.listdir(input_directory):
        if filename.endswith(".txt"):
            print("FILE NAME: "+str(filename))
            with open(str(input_directory)+str("/")+filename, 'r') as file:
                content = file.readlines()

            #Check if txt has not only Killed message
            if len(content) > 1:

                csv_data = []
                found_pattern = False

                for i, line in enumerate(content):
                    if phrase_pattern in line:
                        found_pattern = True
                        continue

                    if i + 2 < len(content):
                        next_line = content[i + 1]
                        after_next_line = content[i + 2]
                        if "Killed" in line or (next_line == "\n" and after_next_line == "\n"):
                            break

                    if found_pattern:
                        row = line.strip().split(',')
                        row = [column.strip() for column in row]  # Remove leading/trailing spaces
                        csv_data.append(row)

                for row in csv_data:
                    row[0] = "type total"
                    row.insert(1, "ops")
                    break

                # Transform the csv into a pandas dataframe for change time columns
                df = pd.DataFrame(csv_data, columns=csv_data[0])
                df = df.iloc[1:]

                df['time'] = pd.to_numeric(df['time'])

                filename = filename.split('_', 1)[0] # To get the timestamp from file name
                num_rows = len(df) # Number de seconds to subtract
                timestamp_dt = datetime.fromtimestamp(float(filename))


                # Iterate over the DataFrame rows
                for index, row in df.iterrows():
                    # Perform your calculation for each row

                    new_timestamp_dt = timestamp_dt - timedelta(seconds=num_rows)
                    new_timestamp_str = new_timestamp_dt.strftime('%Y-%m-%d %H:%M:%S')

                    # Update the value in the 'time' column for the current row
                    df.at[index, 'time'] = new_timestamp_str

                    num_rows = num_rows - 1


                write_to_csv(df, str(output_directory)+str("/")+filename+str(".csv"))
                amout_files = amout_files + 1
    print("CSV files created successfully: "+str(amout_files))


def apply_mean_by_date(input_file):
    # Carregar o arquivo CSV
    df = pd.read_csv(input_file)

    # Converter as colunas de data para o tipo datetime
    df['time'] = pd.to_datetime(df['time'])

    # Identificar as colunas numéricas
    colunas_numericas = df.select_dtypes(include='number').columns

    # Agrupar por data e calcular a média das colunas numéricas
    df_agrupado = df.groupby('time')[colunas_numericas].mean()

    # Reunir o DataFrame agrupado com a coluna textual mantida
    df_final = df_agrupado.join(df['type total'])

    df_final['type total'] = df_final['type total'].fillna('total')


    df_final.to_csv('cassandra-stress.csv', index=True)

def combine_csvs():
    # Directory containing the CSV files
    directory = './csvs'

    # Get a list of all CSV files in the directory
    csv_files = [file for file in os.listdir(directory) if file.endswith('.csv')]

    # Create an empty DataFrame to store the combined data
    combined_data = pd.DataFrame()

    # Loop through each CSV file and append its data to the combined DataFrame
    for file in csv_files:
        file_path = os.path.join(directory, file)
        df = pd.read_csv(file_path)
        combined_data = combined_data.append(df, ignore_index=True)

    # Output file path for the combined CSV
    output_file = './combined.csv'

    # Save the combined DataFrame to a CSV file
    combined_data.to_csv(output_file, index=False)

    print("CSV files combined successfully.")

phrase_pattern = "Failed to connect over JMX; not collecting these stats"
input_directory = "./txts/"
output_directory = "./csvs"

csv_pattern = extract_csv_pattern(input_directory, phrase_pattern, output_directory)
combine_csvs()
apply_mean_by_date("combined.csv")






