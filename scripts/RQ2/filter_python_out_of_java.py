import os
import json
import shutil


def list_json_files(root_dir):
    """ List all JSON files in the specified directory and its subdirectories. """
    print(f"Searching in directory: {root_dir}")
    for dirpath, dirnames, filenames in os.walk(root_dir):
        print(f"Checking directory: {dirpath}")
        for filename in filenames:
            if filename.endswith('.json'):
                json_file_path = os.path.join(dirpath, filename)
                print(f"Found JSON file: {json_file_path}")
                yield json_file_path


def is_python_file(filename):
    """ Check if the given filename is a Python file. """
    return filename.endswith('.py')


def copy_json_if_python_files(json_file, dest_dir):
    """ Copy the JSON file to the destination directory if it contains Python files. """
    with open(json_file, 'r') as file:
        data = json.load(file)
        updated_files = data.get('Updated_Files', [])

        print(f"Updated Files in {json_file}: {updated_files}")

        if any(is_python_file(f) for f in updated_files):
            print(f"Python file found in {json_file}. Copying to destination.")
            rel_path = os.path.relpath(json_file, start=source_dir)
            dest_path = os.path.join(dest_dir, rel_path)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            shutil.copyfile(json_file, dest_path)


source_dir = './Data_Java_Python'
dest_dir = 'new_destination_folder'

os.makedirs(dest_dir, exist_ok=True)

for json_file in list_json_files(source_dir):
    copy_json_if_python_files(json_file, dest_dir)
