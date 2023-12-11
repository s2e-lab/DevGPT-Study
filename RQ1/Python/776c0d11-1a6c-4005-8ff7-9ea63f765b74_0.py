import pandas as pd

# Sample DataFrame
data = {'date': ['2023-06-01', '2023-06-01', '2023-06-02', '2023-06-02', '2023-06-03'],
        'value': [10, 20, 30, 40, 50]}
df = pd.DataFrame(data)

# Convert the 'date' column to datetime type
df['date'] = pd.to_datetime(df['date'])

# Group the DataFrame by the 'date' column
grouped = df.groupby(pd.Grouper(key='date', freq='D'))

# Iterate over the groups and create individual DataFrames
for group_name, group_data in grouped:
    # Create a new DataFrame for each group
    new_df = pd.DataFrame(group_data)
    
    # Perform any additional operations or analysis on the new DataFrame
    
    # Print the new DataFrame for demonstration purposes
    print(f"DataFrame for {group_name}:")
    print(new_df)
    print()
