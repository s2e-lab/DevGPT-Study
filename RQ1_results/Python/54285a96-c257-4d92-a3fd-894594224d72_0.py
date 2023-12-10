import os
import pandas as pd

def process_files(directory):
    # Initialize a dictionary to store data for each expenditure type
    expenditure_data = {}

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.xls'):
                full_path = os.path.join(root, file)

                # Read the Excel file into a DataFrame
                df = pd.read_excel(full_path)

                # Extract the year from the directory name
                year = os.path.basename(root)

                # Find the columns with the expenditure types
                expenditure_cols = df.columns[df.iloc[6] == "Services"].tolist()

                # Loop through each expenditure type column
                for col in expenditure_cols:
                    expenditure_type = df.iloc[5][col]

                    # Filter rows where the expenditure type is "Per Funded Pupil Count"
                    filtered_df = df[df.iloc[:, 3] == "Per Funded Pupil Count"]

                    # Create a unique key for the expenditure type
                    key = f"{year}_{expenditure_type}"

                    # Get the school districts and corresponding expenditure values
                    districts = filtered_df.iloc[:, 1].values
                    expenditures = filtered_df.iloc[:, col].values

                    # Store the data in the dictionary
                    if key not in expenditure_data:
                        expenditure_data[key] = {}
                    expenditure_data[key][year] = dict(zip(districts, expenditures))

    return expenditure_data

def export_to_csv(expenditure_data):
    for expenditure_type, data in expenditure_data.items():
        # Create a DataFrame with the data
        df = pd.DataFrame(data)

        # Export the DataFrame to a CSV file
        output_file = f"{expenditure_type}.csv"
        df.to_csv(output_file)

if __name__ == "__main__":
    input_directory = "/path/to/your/input/directory"
    expenditure_data = process_files(input_directory)
    export_to_csv(expenditure_data)
