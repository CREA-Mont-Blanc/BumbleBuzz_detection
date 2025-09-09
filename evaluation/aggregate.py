### open all csv files which begins with metrics_ and concatenate them
import os
import pandas as pd
import glob
import argparse

def aggregate_metrics(input_dir, output_file):
    # Get all CSV files in the input directory that finish with '.csv'
    
    csv_files = glob.glob(os.path.join(input_dir, '*.csv'))
    ## keep only the ones finishing with _metrics.csv
    csv_files = [f for f in csv_files if os.path.basename(f).endswith('_metrics.csv')]

    # Initialize an empty list to store DataFrames
    dataframes = []

    # Loop through each CSV file and read it into a DataFrame
    for file in csv_files:
        df = pd.read_csv(file)
        # add the site name as a new column
        site_name = os.path.basename(file).split('_')[1]
        df['site'] = site_name
        # add the file name as a new column
        df['file'] = os.path.basename(file)
        ## put these columns at the beginning
        cols = df.columns.tolist()
        cols = cols[-2:] + cols[:-2]
        df = df[cols]
        dataframes.append(df)

    # Concatenate all DataFrames into a single DataFrame
    combined_df = pd.concat(dataframes, ignore_index=True)

    # Save the combined DataFrame to the output file
    combined_df.to_csv(output_file, index=False)
    print(f"Aggregated metrics saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate metrics from multiple CSV files.")
    parser.add_argument("input_dir", type=str, help="Directory containing the CSV files.")
    parser.add_argument("output_file", type=str, help="Output file for aggregated metrics.")
    
    args = parser.parse_args()
    
    aggregate_metrics(args.input_dir, args.output_file)
