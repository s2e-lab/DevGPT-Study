import json
import base64
import requests
import os
import glob
import time
from datetime import datetime

# NOTE: This program collected the complete commit tree with complete files for each PR

with open('../config.json', 'r') as file:
    github_token = json.load(file)['GITHUB_TOKEN']

headers = {'Authorization': f'token {github_token}'}


def check_rate_limit():
    rate_limit_url = 'https://api.github.com/rate_limit'
    response = requests.get(rate_limit_url, headers=headers).json()
    limit = response['resources']['core']['limit']
    remaining = response['resources']['core']['remaining']
    reset_time = response['resources']['core']['reset']
    return remaining, limit, reset_time


def wait_for_rate_limit_reset(reset_time):
    current_time = time.time()
    reset_time = datetime.fromtimestamp(reset_time)
    print(f"Rate limit exceeded. Waiting for reset at {reset_time}")
    time.sleep(max(reset_time - current_time, 0))


def fetch_file_content(url):
    remaining, _, reset_time = check_rate_limit()
    if remaining < 10:
        wait_for_rate_limit_reset(reset_time)

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        content = response.json()['content']
        return base64.b64decode(content).decode('utf-8')
    except requests.exceptions.HTTPError as err:
        if err.response.status_code == 403:
            wait_for_rate_limit_reset(reset_time)
            return fetch_file_content(url)
        print(f"HTTP error occurred: {err}")
    except requests.exceptions.Timeout as err:
        print("Request timed out, retrying...")
        return fetch_file_content(url)
    except Exception as err:
        print(f"An error occurred: {err}")
    return None


def ensure_directory_exists(file_path):
    try:
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
    except Exception as err:
        print(f"Error creating directory {directory}: {err}")


json_files = glob.glob('../Merged_Unique_PR_Sharings/*.json')

for json_file in json_files:
    try:
        with open(json_file, 'r') as file:
            data = json.load(file)

        updated_files = data["Updated_Files"]
        tree = data["Commit_Tree"]["tree"]

        file_info = {f["path"]: f["url"]
                     for f in tree if f["path"] in updated_files}

        output_folder = os.path.join(
            '../output', os.path.basename(json_file).split('.')[0])
        ensure_directory_exists(output_folder)

        for file_path, url in file_info.items():
            full_path = os.path.join(output_folder, file_path)
            ensure_directory_exists(full_path)
            content = fetch_file_content(url)
            if content:
                with open(full_path, 'w') as file:
                    file.write(content)
                    print(f"File {full_path} written successfully.")
    except json.JSONDecodeError as err:
        print(f"JSON decoding error in file {json_file}: {err}")
    except Exception as err:
        print(f"Error processing file {json_file}: {err}")
