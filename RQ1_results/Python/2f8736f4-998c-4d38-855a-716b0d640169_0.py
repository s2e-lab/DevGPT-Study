import json

def filter_json(input_file, output_file1, output_file2, pid_list):
    # Read the input JSON file
    with open(input_file, 'r') as f:
        data = json.load(f)

    matching_entries = []  # List to store matching entries
    non_matching_entries = []  # List to store non-matching entries

    # Iterate over each entry in the input data
    for entry in data:
        if 'pid' in entry and entry['pid'] in pid_list:
            # Entry has a "pid" field and its value is in the specified pid_list
            matching_entries.append(entry)
        else:
            # Entry does not have a "pid" field or its value is not in the specified pid_list
            non_matching_entries.append(entry)

    # Write the matching entries to output_file1 as JSON
    with open(output_file1, 'w') as f1:
        json.dump(matching_entries, f1, indent=4)

    # Write the non-matching entries to output_file2 as JSON
    with open(output_file2, 'w') as f2:
        json.dump(non_matching_entries, f2, indent=4)

    print("Filtered data saved successfully!")

# Example usage
input_file = 'data.json'  # Path to the input JSON file
output_file1 = 'matching_entries.json'  # Path to save the matching entries JSON file
output_file2 = 'non_matching_entries.json'  # Path to save the non-matching entries JSON file
pid_list = [1, 3, 5]  # List of pids to match

filter_json(input_file, output_file1, output_file2, pid_list)
