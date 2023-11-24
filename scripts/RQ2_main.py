import json
import requests
import os
import shutil
import glob
import time
import logging

logging.basicConfig(filename='error.log',
                    filemode='a',
                    format='%(asctime)s - %(message)s',
                    level=logging.INFO)

# NOTE: This program contain most logic in RQ2 data collection
# It will fetch the commit tree and the list of updated files for each PR
# and save them to a file in the Data directory


def load_github_token():
    with open('./config.json', 'r') as file:
        config_data = json.load(file)
    return config_data['GITHUB_TOKEN']


def check_rate_limit(response):
    # Extract rate limit info from headers
    remaining = int(response.headers.get('X-RateLimit-Remaining', 0))
    reset_time = int(response.headers.get('X-RateLimit-Reset', 0))

    # If we are approaching the rate limit, wait until the reset time
    if remaining < 10:  # wait if fewer than 10 requests are remaining
        wait_duration = reset_time - int(time.time())
        if wait_duration > 0:
            print(
                f"Approaching rate limit. Waiting for {wait_duration} seconds.")
            time.sleep(wait_duration)


def get_pr_metadata(repo_name, pr_number, token):
    url = f"https://api.github.com/repos/{repo_name}/pulls/{pr_number}"
    headers = {'Authorization': f'token {token}'}
    response = requests.get(url, headers=headers)
    check_rate_limit(response)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching PR metadata: {response.status_code}")
        logging.error(
            f"Error fetching PR metadata: {response.status_code}, {url}")
        return None


def get_commit_tree(repo_name, commit_sha, token):
    url = f"https://api.github.com/repos/{repo_name}/git/trees/{commit_sha}?recursive=1"
    headers = {'Authorization': f'token {token}'}
    response = requests.get(url, headers=headers)
    check_rate_limit(response)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching commit tree: {response.status_code}")
        logging.error(
            f"Error fetching PR metadata: {response.status_code}, {url}")
        return None


def get_pr_commits(repo_name, pr_number, token):
    url = f"https://api.github.com/repos/{repo_name}/pulls/{pr_number}/commits"
    headers = {'Authorization': f'token {token}'}
    response = requests.get(url, headers=headers)
    check_rate_limit(response)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching PR commits: {response.status_code}")
        logging.error(
            f"Error fetching PR metadata: {response.status_code}, {url}")
        return []


# Function to get detailed file changes for a commit
def get_commit_changes(repo_name, commit_sha, token):
    url = f"https://api.github.com/repos/{repo_name}/commits/{commit_sha}"
    headers = {'Authorization': f'token {token}'}
    response = requests.get(url, headers=headers)
    check_rate_limit(response)
    if response.status_code == 200:
        commit_data = response.json()
        return commit_data.get('files', [])
    else:
        print(f"Error fetching commit changes: {response.status_code}")
        return []


def save_tree_and_changes_to_file(tree, updated_files, repo_name, pr_number, pr_url, output_directory):
    sanitized_repo_name = repo_name.replace('/', '_')
    file_name = f"{output_directory}/{sanitized_repo_name}_PR_{pr_number}_commit_tree.json"

    data_to_save = {
        "PR_URL": pr_url,
        "Commit_Tree": tree,
        "Updated_Files": updated_files
    }

    with open(file_name, 'w') as file:
        json.dump(data_to_save, file, indent=4)
    print(f"Data for PR {pr_number} in repo {repo_name} saved to {file_name}")


def process_pull_requests(json_file_path, github_token):
    with open(json_file_path, 'r') as file:
        pr_data = json.load(file)

    processed_prs = set()  # Set to track processed PRs

    for pr in pr_data:
        pr_repo_name = pr['RepoName']
        pr_number = pr['Number']
        pr_url = pr['URL']
        # Unique identifier for each PR
        pr_identifier = f"{pr_repo_name}_{pr_number}"

        if pr_identifier in processed_prs:
            print(
                f"Duplicate PR detected: {pr_repo_name} PR {pr_number}. Skipping.")
            logging.error(
                f"Duplicate PR detected: {pr_repo_name} PR {pr_number}. Skipping.")
            continue
        else:
            processed_prs.add(pr_identifier)

        pr_metadata = get_pr_metadata(pr_repo_name, pr_number, github_token)
        if pr_metadata and 'merge_commit_sha' in pr_metadata:
            merge_commit_sha = pr_metadata['merge_commit_sha']
            commit_tree = get_commit_tree(
                pr_repo_name, merge_commit_sha, github_token)

            pr_commits = get_pr_commits(pr_repo_name, pr_number, github_token)
            updated_files = set()

            for commit in pr_commits:
                commit_sha = commit.get('sha')
                commit_changes = get_commit_changes(
                    pr_repo_name, commit_sha, github_token)
                for file_change in commit_changes:
                    updated_files.add(file_change['filename'])

            if commit_tree and 'tree' in commit_tree:
                save_tree_and_changes_to_file(commit_tree, list(
                    updated_files), pr_repo_name, pr_number, pr_url, data_directory)
            else:
                print(
                    f"No commit tree found for PR {pr_number} in repo {pr_repo_name}")


pattern = 'snapshot_*/*_pr_sharings_merged.json'
github_token = load_github_token()

for json_file_path in glob.glob(pattern):
    snapshot_dir = os.path.dirname(json_file_path)
    print(f'Processing PRs from {snapshot_dir}')
    data_directory = os.path.join(snapshot_dir, "Data")
    # Clear out the Data directory if it exists, then recreate it
    if os.path.exists(data_directory):
        shutil.rmtree(data_directory)
    os.makedirs(data_directory)
    print(f'Data will be saved in {data_directory}')

    process_pull_requests(json_file_path, github_token)
    print(f'Processed PRs from {json_file_path} and saved in {data_directory}')
