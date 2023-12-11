import requests

def check_commits_difference(owner, repo, branch_a, branch_b, access_token=None):
    base_url = "https://api.github.com"
    headers = {"Authorization": f"token {access_token}"} if access_token else {}
    compare_url = f"{base_url}/repos/{owner}/{repo}/compare/{branch_a}...{branch_b}"
    
    response = requests.get(compare_url, headers=headers)

    if response.status_code == 200:
        comparison_data = response.json()
        ahead_by = comparison_data.get("ahead_by", 0)
        return ahead_by > 0
    else:
        print(f"Failed to fetch comparison data. Status code: {response.status_code}")
        return None

# Replace these values with your actual data
owner = "your_username"
repo = "your_repository"
branch_a = "A"
branch_b = "B"
access_token = "your_github_access_token"  # Only required for private repositories

result = check_commits_difference(owner, repo, branch_a, branch_b, access_token)
if result is not None:
    if result:
        print("Branch B has commits that branch A does not.")
    else:
        print("Branch B does not have any commits that branch A does not.")
