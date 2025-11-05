import os
import glob
import pandas as pd

def combine_csvs_from_folder(input_dir, output_file, sort_columns=None):
    """
    Combine all CSV files in the specified folder into a single CSV file.

    Parameters:
    - input_dir (str): Folder containing the CSV files.
    - output_file (str): Path to the output combined CSV file.
    - sort_columns (list of str, optional): Columns to sort by before saving.
    """
    if not os.path.isdir(input_dir):
        print(f"Error: Directory not found at {input_dir}")
        return

    csv_files = glob.glob(os.path.join(input_dir, '*.csv'))
    print(f"Found {len(csv_files)} CSV files in {input_dir}")

    if not csv_files:
        print("No CSV files found. Exiting.")
        return

    combined_df = pd.DataFrame()

    for file in csv_files:
        print(f"Reading {file}")
        try:
            df = pd.read_csv(file)
            combined_df = pd.concat([combined_df, df], ignore_index=True)
        except Exception as e:
            print(f"Error reading {file}: {e}")

    if combined_df.empty:
        print("No data to write. Combined DataFrame is empty.")
        return

    if sort_columns:
        missing_cols = [col for col in sort_columns if col not in combined_df.columns]
        if missing_cols:
            print(f"Warning: Some sort columns not found in DataFrame: {missing_cols}")
        else:
            # Check for 'date' column to convert to datetime for proper sorting
            date_col_to_convert = None
            for col in ['date', 'Date']: # Check for common date column names
                if col in sort_columns and col in combined_df.columns:
                    date_col_to_convert = col
                    break
            
            if date_col_to_convert:
                try:
                    # Convert column to datetime objects
                    combined_df[date_col_to_convert] = pd.to_datetime(combined_df[date_col_to_convert])
                    print(f"Converted '{date_col_to_convert}' to datetime for sorting.")
                except Exception as e:
                    print(f"Warning: Could not convert date column '{date_col_to_convert}' to datetime: {e}. Sorting will be string-based.")

            print(f"Sorting by {sort_columns}...")
            combined_df = combined_df.sort_values(sort_columns)

    combined_df.to_csv(output_file, index=False)
    print(f"Combined CSV saved to {output_file}")

if __name__ == "__main__":
    # ============
    # Breakfast
    # ============
    input_dir = '../data/preprocessed-data/Breakfast production'
    output_file = '../data/preprocessed-data/breakfast_combined.csv'
    sort_columns = ['school_name', 'date', 'identifier']
    combine_csvs_from_folder(input_dir, output_file, sort_columns)

    # ============
    # Lunch
    # ============
    input_dir = '../data/preprocessed-data/Lunch production'
    output_file = '../data/preprocessed-data/lunch_combined.csv'
    sort_columns = ['school_name', 'date', 'identifier']
    combine_csvs_from_folder(input_dir, output_file, sort_columns)