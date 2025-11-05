import pandas as pd
import re
import os
import glob
from bs4 import BeautifulSoup

def parse_school_table(school_name, table, date):
    """
    Parses an HTML table for a single school and returns a DataFrame
    with a predefined, flattened column structure.
    """
    print(f"Parsing table for school: {school_name}")

    # Define the correct, flattened column names for the final DataFrame.
    final_columns = [
        "school_name", "date", "identifier", "name", "planned_reimbursable",
        "planned_non-reimbursable", "planned_total", "offered", "served_reimbursable", 
        "served_non-reimbursable", "served_total", "served_cost", "discarded_total", 
        "discarded_percent_of_offered", "discarded_cost", "subtotal_cost", "left_over_total", 
        "left_over_percent_of_offered", "left_over_cost", "production_cost_total"
    ]
    # Number of expected data columns in the HTML table (total columns - 2)
    num_data_columns = len(final_columns) - 2

    rows = table.find('tbody').find_all('tr')
    data = []

    for row in rows:
        # Skip footer rows which are sometimes used for totals
        if row.get('class') and 'footer' in row.get('class'):
            continue

        # Extract all data cells from the current row
        cells = [c.get_text(strip=True) for c in row.find_all('td')]

        # Ensure the row has the expected number of columns before processing
        if len(cells) >= num_data_columns:
            # Prepend school name and date to the first N data cells
            record = [school_name, date] + cells[:num_data_columns]
            data.append(record)
        elif cells: # Log a warning for non-empty but malformed rows
            print(f"  -> Warning: Skipping row with {len(cells)} cells for {school_name}, expected {num_data_columns}.")

    if not data:
        print(f"No valid data rows found for school: {school_name}")
        return None

    # Create the DataFrame with the predefined, correct column names
    df = pd.DataFrame(data, columns=final_columns)
    print(f"Created DataFrame for {school_name} with {len(df)} rows and {len(df.columns)} columns")

    return df

def parse_html_file(file_path):
    print(f"Processing file: {file_path}")

    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return []

    # Read the HTML file
    with open(file_path, 'r', encoding='utf-8') as f:
        html_data = f.read()

    # Parse HTML with BeautifulSoup
    soup = BeautifulSoup(html_data, 'html.parser')

    # Try to find the filters section
    filters_section = soup.find(string=re.compile(r'Date Range', re.I))
    date = 'Unknown'

    if filters_section:
        print(f"Filters section found: {filters_section[:100]}...")
        # Try multiple date patterns
        date_patterns = [
            r'Date Range\s*\(Start = (\d+/\d+/\d+), End = \d+/\d+/\d+\)',  # MM/DD/YYYY
            r'Date Range\s*\(Start = (\d+-\d+-\d+), End = \d+-\d+-\d+\)',  # MM-DD-YYYY
            r'Date Range\s*\(Start = ([A-Za-z]+ \d+, \d{4}), End = [A-Za-z]+ \d+, \d{4}\)',  # Month DD, YYYY
            r'Date Range\s*:\s*(\d+/\d+/\d+)',  # Single date MM/DD/YYYY
            r'Date Range\s*:\s*(\d+-\d+-\d+)'  # Single date MM-DD-YYYY
        ]
        for pattern in date_patterns:
            match = re.search(pattern, html_data, re.I)
            if match:
                date = match.group(1)
                break
    else:
        print(f"No filters section found in {file_path}")

    # Fallback: Try to infer date from file name (e.g., 5.01.25 breakfast.html → 5/1/2025)
    if date == 'Unknown':
        file_name = os.path.basename(file_path)
        date_match = re.search(r'(\d+)\.(\d+)\.(\d+)', file_name)
        if date_match:
            month, day, year = date_match.groups()
            year = f"20{year}" if len(year) == 2 else year  # Convert YY to YYYY
            date = f"{month}/{day}/{year}"
            print(f"Inferred date from file name: {date}")

    print(f"Using date: {date}")

    # Find all page-break divs (each contains a school)
    page_breaks = soup.find_all('div', class_='page-break')
    print(f"Found {len(page_breaks)} school sections in {file_path}")

    # Initialize list to store DataFrames for this file
    file_dfs = []

    # Process each school section
    for page_break in page_breaks:
        # Extract school name
        school_name_elem = page_break.find('div', class_='sub-heading').find('li')
        school_name = school_name_elem.text.strip() if school_name_elem else 'Unknown School'
        print(f"Processing school: {school_name}")

        # Extract table
        table = page_break.find('table', class_='striped')
        if table:
            df = parse_school_table(school_name, table, date)
            if df is not None:
                file_dfs.append(df)
        else:
            print(f"No table found for school: {school_name}")

    return file_dfs

def generate_csvs_from_folder(folder_path, output_dir='output_csvs'):
    # Check if folder exists
    if not os.path.isdir(folder_path):
        print(f"Error: Folder not found at {folder_path}")
        return

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Get all HTML files in the folder
    html_files = glob.glob(os.path.join(folder_path, '*.html'))
    print(f"Found {len(html_files)} HTML files in {folder_path}")

    if not html_files:
        print("Error: No HTML files found in the folder.")
        return

    # Process each HTML file
    for file_path in html_files:
        # Get the base name of the file and create CSV name
        file_name = os.path.basename(file_path)
        csv_name = os.path.splitext(file_name)[0] + '.csv'
        output_file = os.path.join(output_dir, csv_name)
        print(f"Generating CSV: {output_file}")

        # Parse the HTML file
        file_dfs = parse_html_file(file_path)

        if not file_dfs:
            print(f"No valid data found for {file_name}. Skipping CSV generation.")
            continue

        # Combine DataFrames for this file
        final_df = pd.concat(file_dfs, ignore_index=True)

        # Sort by School_Name, Date, Identifier
        if 'Identifier' in final_df.columns:
            final_df = final_df.sort_values(['School_Name', 'Date', 'Identifier'])

        # Save to CSV
        final_df.to_csv(output_file, index=False)
        print(f"CSV file generated: {output_file}")

if __name__ == "__main__":
    # NOTE: You may need to update these paths to match your folder structure
    folder_path_lunch = "../data/FairfaxCounty/May 2025 Lunch production records/May 2025 Lunch production records"
    output_dir_lunch = "../data/preprocessed-data/Lunch production"
    generate_csvs_from_folder(folder_path_lunch, output_dir_lunch)

    folder_path_breakfast = "../data/FairfaxCounty/May 2025 Breakfast production records/May 2025 Breakfast production records"
    output_dir_breakfast = "../data/preprocessed-data/Breakfast production"
    generate_csvs_from_folder(folder_path_breakfast, output_dir_breakfast)