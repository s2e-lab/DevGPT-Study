import json
import glob
import os

# NOTE: Check if the PR was merged

pattern = 'snapshot_*/*_pr_sharings.json'

for filename in glob.glob(pattern):
    with open(filename, 'r') as file:
        data = json.load(file)

    filtered_prs = []

    for pr in data["Sources"]:
        if pr["MergedAt"] is not None:
            filtered_prs.append(pr)

    new_file_name = os.path.splitext(filename)[0] + '_merged.json'

    with open(new_file_name, 'w') as new_file:
        json.dump(filtered_prs, new_file, indent=4)

    print(f'Filtered PRs written to {new_file_name}')
