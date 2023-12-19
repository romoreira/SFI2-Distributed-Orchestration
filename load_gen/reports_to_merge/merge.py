import csv
import os
import pandas as pd
from datetime import datetime, timedelta
import time
import numpy as np

working_dir = "./read_sinusoidal"
phrase_pattern = "Failed to connect over JMX; not collecting these stats"
input_directory = working_dir+"/txts/"
output_directory = working_dir+"/csvs/"

def write_to_csv(df, output_file):
    df.to_csv(output_file, index=False)
def extract_csv_pattern(input_directory, phrase_pattern, output_directory):
    print("Extracting CSVs from TXTs...")
    amount_files = 0
    for filename in os.listdir(input_directory):
        if filename.endswith(".txt"):
            #print("FILE NAME: "+str(filename))
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
                amount_files = amount_files + 1
    print(f"TXTs converted to CSV successfully: "+str(amount_files)+str(" at: "),{str(output_directory)})


def apply_mean_by_date(input_file):
    arquivo_flows = os.path.join(working_dir, './cassandra_stress.csv')
    if os.path.exists(arquivo_flows):
        # Remove o arquivo "flows.csv" caso ele exista
        os.remove(arquivo_flows)
    else:
        print("File not found. cassandra_stress.csv")
        print("Applying mean by date into cassandra-stress CSV..")


    # Carregar o arquivo CSV
    df = pd.read_csv(working_dir+str("/")+input_file)


    # Identificar as colunas numéricas
    colunas_numericas = df.select_dtypes(include='number').columns

    # Agrupar por data e calcular a média das colunas numéricas
    df_agrupado = df.groupby('time')[colunas_numericas].mean()

    # Reunir o DataFrame agrupado com a coluna textual mantida
    df_final = df_agrupado.join(df['type total'])


    df_final['type total'] = df_final['type total'].fillna('total')

    df_final.to_csv(working_dir+'/cassandra_stress.csv', index=False)
    print(f"Mean applied successfully in cassandra-stress CSV at:",{working_dir+'/cassandra_stress.csv'})

def shift_three_hours(df):

    # Converter a coluna de timestamps para datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')  # 's' representa segundos

    # Subtrair três horas da coluna de timestamps
    df['timestamp'] = df['timestamp'] - pd.Timedelta(hours=3)
    df['timestamp'] = (df['timestamp'] - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)

    return df

def merge_csvs(cassandra, netflow):
    # Leitura dos arquivos CSV
    df1 = pd.read_csv(working_dir+"/"+cassandra)
    df2 = pd.read_csv(working_dir+"/"+netflow)

    #Change column name
    df2 = df2.rename(columns={'timestamp': 'time'})

    # Mescla dos dataframes com base na coluna 'data_hora'
    merged_df = pd.merge(df1, df2, on='time', how='inner')

    # Visualizar o resultado
    #print(merged_df)

    # Caso queira salvar o resultado em um novo arquivo CSV
    merged_df.to_csv(working_dir+'/arquivo_final.csv', index=False)
    print("Files merged successfully.")

def combine_csvs():
    arquivo_flows = os.path.join(working_dir, './cassandra-stress_combined.csv')
    if os.path.exists(arquivo_flows):
        # Remove o arquivo "flows.csv" caso ele exista
        os.remove(arquivo_flows)
    else:
        print("File not found. cassandra-stress_combined.csv")
        print("Combining CSVs from cassandra-stress..")

    # Directory containing the CSV files
    directory = working_dir+"/csvs"

    # Get a list of all CSV files in the directory
    csv_files = [file for file in os.listdir(directory) if file.endswith('.csv')]

    # Create an empty DataFrame to store the combined data
    combined_data = pd.DataFrame()

    # Loop through each CSV file and append its data to the combined DataFrame
    for file in csv_files:
        file_path = os.path.join(directory, file)
        df = pd.read_csv(file_path)
        combined_data = combined_data.append(df, ignore_index=True)

    combined_data['time'] = pd.to_datetime(combined_data['time']).astype(int) // 10 ** 9

    # Output file path for the combined CSV
    output_file = working_dir+'/cassandra-stress_combined.csv'



    # Save the combined DataFrame to a CSV file
    combined_data.to_csv(output_file, index=False)

    print(f"CSV of cassandra-stress combined successfully at:",{output_file})

def convert_to_unix_epoch(timestamp_str):
    timestamp_obj = time.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
    unix_epoch = int(time.mktime(timestamp_obj))
    return unix_epoch

def combine_flows_csvs():
    arquivo_flows = os.path.join(working_dir, './cassandra_flows.csv')
    if os.path.exists(arquivo_flows):
        # Remove o arquivo "flows.csv" caso ele exista
        os.remove(arquivo_flows)
    else:
        print("File not found. cassandra_flows.csv")
        print("Creating file..")


    print("Combining CSVs from Cassandra Flows..")
    dataframes_list = []
    merged_df = ""
    for filename in os.listdir(working_dir+'/csvs/cassandra_flows'):
        if filename.endswith('.csv'):
            file_path = os.path.join(working_dir+'/csvs/cassandra_flows', filename)
            df = pd.read_csv(file_path)
            dataframes_list.append(df)

    if len(dataframes_list) > 0:
        merged_df = dataframes_list[0]

        for df in dataframes_list[1:]:
            merged_df = pd.concat([merged_df, df.loc[df['timestamp'] != merged_df['timestamp'].iloc[-1]]])
    else:
        return None


    output_file = working_dir+'/cassandra_flows.csv'
    #merged_df['Timestamp'] = merged_df['Timestamp'].apply(convert_to_unix_epoch)
    columns=['Src IP', 'Src Port', 'Dst IP', 'Dst Port', 'Label']
    merged_df = merged_df.groupby('timestamp').mean().reset_index()
    merged_df[columns] = np.nan
    merged_df['timestamp'] = pd.to_datetime(merged_df['timestamp']).astype(int) // 10 ** 9
    merged_df = shift_three_hours(merged_df)
    merged_df.to_csv(output_file, index=False)
    print(f"Final file (cassandra_flows.csv) created at: {output_file}")

combine_flows_csvs()
extract_csv_pattern(input_directory, phrase_pattern, output_directory)
combine_csvs()
apply_mean_by_date("cassandra-stress_combined.csv")
merge_csvs("cassandra_stress.csv","cassandra_flows.csv")





