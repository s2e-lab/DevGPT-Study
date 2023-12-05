import os
import json


def traverse_and_extract_pr_urls(base_path):
    base_dir = os.path.join(base_path, 'results')
    if not os.path.exists(base_dir):
        return "Base directory does not exist"
    pr_urls = []
    for folder in os.listdir(base_dir):
        if folder.startswith("Data_Java_Python_"):
            folder_path = os.path.join(base_dir, folder)
            for file in os.listdir(folder_path):
                if file.endswith(".json"):
                    file_path = os.path.join(folder_path, file)
                    with open(file_path, 'r') as json_file:
                        try:
                            data = json.load(json_file)
                            pr_url = data.get("PR_URL")
                            if pr_url:
                                pr_urls.append(pr_url)
                        except json.JSONDecodeError:
                            print("JSONDecodeError: " + file_path)
                            continue

    output_file_path = os.path.join(base_path, "pr_urls.txt")
    with open(output_file_path, 'w') as output_file:
        for url in pr_urls:
            output_file.write(url + "\n")

    return "PR URLs extracted and saved to pr_urls.txt"


result = traverse_and_extract_pr_urls('../')
print(result)
