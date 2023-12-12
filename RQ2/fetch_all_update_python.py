import os
import json

folder_path = 'Data_Python/Data_Python_20231012'
output_file = 'python_files.txt'


def extract_python_files(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
        updated_files = data.get('Updated_Files', [])
        python_files = [file for file in updated_files if file.endswith('.py')]
        return python_files


with open(output_file, 'w') as out_file:
    for file in os.listdir(folder_path):
        if file.endswith('.json'):
            file_path = os.path.join(folder_path, file)
            out_file.write(file + '\n\n')
            python_files = extract_python_files(file_path)
            for python_file in python_files:
                out_file.write(python_file + '\n')
            out_file.write('\n')


print(f"The list of Python files has been written to {output_file}")
