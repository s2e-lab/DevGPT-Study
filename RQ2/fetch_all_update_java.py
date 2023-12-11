import os
import json

folder_path = 'Data_Java/Data_Java_Python_20231012'
output_file = 'java_files.txt'


def extract_java_files(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
        updated_files = data.get('Updated_Files', [])
        java_files = [file for file in updated_files if file.endswith('.java')]
        return java_files


with open(output_file, 'w') as out_file:
    for file in os.listdir(folder_path):
        if file.endswith('.json'):
            file_path = os.path.join(folder_path, file)
            out_file.write(file + '\n\n')
            java_files = extract_java_files(file_path)
            for java_file in java_files:
                out_file.write(java_file + '\n')
            out_file.write('\n')


print(f"The list of Java files has been written to {output_file}")
