import csv
import os
import pandas as pd
from datetime import datetime, timedelta

def write_to_csv(df, output_file):
    df.to_csv(output_file, index=False)
def extract_csv_pattern(file_path, phrase_pattern, input_directory):
    for filename in os.listdir(input_directory):
        if filename.endswith(".txt"):
            with open(filename, 'r') as file:
                content = file.readlines()

            csv_data = []
            found_pattern = False

            for line in content:
                if phrase_pattern in line:
                    found_pattern = True
                    continue

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


            write_to_csv(df, filename+str(".csv"))



# Usage example
file_path = '1689731503.457156_output_stdout.txt'  # Replace with the actual file path
phrase_pattern = "Failed to connect over JMX; not collecting these stats"
input_directory = "."
output_directory = "."

csv_pattern = extract_csv_pattern(input_directory, phrase_pattern, output_directory)







