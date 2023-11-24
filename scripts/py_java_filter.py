import glob
import json
import os
import shutil

# NOTE: Check if updated_files includes .java or .py files


def process_json_files():
    pattern = 'snapshot_*/Data/*.json'

    for json_file in glob.glob(pattern):
        with open(json_file, 'r') as file:
            data = json.load(file)

        if any(file.endswith(('.java', '.py')) for file in data.get('Updated_Files', [])):
            snapshot_dir = os.path.dirname(json_file)
            new_directory = os.path.join(snapshot_dir, "Data_Java_Python")

            if not os.path.exists(new_directory):
                os.makedirs(new_directory)

            new_file_path = os.path.join(
                new_directory, os.path.basename(json_file))

            shutil.copy(json_file, new_file_path)
            print(f"Copied {json_file} to {new_file_path}")


process_json_files()
