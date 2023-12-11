import os
import json
import shutil

def traverse_and_copy_unique_files(base_path):
    base_dir = os.path.join(base_path, 'results')
    output_dir = os.path.join(base_path, "Merged_Unique_PR_Sharings")

    if not os.path.exists(base_dir):
        return "Base directory does not exist"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    existing_files = set()
    copied_files_count = 0

    for folder in os.listdir(base_dir):
        if folder.startswith("Data_Java_Python_"):
            folder_path = os.path.join(base_dir, folder)
            for file in os.listdir(folder_path):
                if file.endswith(".json"):
                    file_path = os.path.join(folder_path, file)
                    try:
                        with open(file_path, 'r') as json_file:
                            data = json.load(json_file)
                            pr_url = data.get("PR_URL")
                            if pr_url:
                                file_identity = (file, pr_url)
                                if file_identity not in existing_files:
                                    existing_files.add(file_identity)
                                    shutil.copy(file_path, os.path.join(output_dir, file))
                                    copied_files_count += 1
                    except json.JSONDecodeError:
                        print("JSONDecodeError: " + file_path)
                        continue

    return f"Unique files copied to {output_dir}. Total files copied: {copied_files_count}"

result = traverse_and_copy_unique_files('../../')
print(result)
