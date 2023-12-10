def analyze_repos(file_path):
    repo_counts = {}

    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split('/')
            # Extract just the repository name
            repo_name = parts[4] if len(parts) > 4 else None
            if repo_name:
                repo_counts[repo_name] = repo_counts.get(repo_name, 0) + 1

    duplicates = [repo for repo, count in repo_counts.items() if count > 1]
    unique_repo_count = len(repo_counts) - len(duplicates)

    return unique_repo_count, duplicates


file_path = 'pr_unique_url_python.txt'
unique_repo_count, duplicate_repos = analyze_repos(file_path)

print(f"Number of unique repositories: {unique_repo_count}")
print(f"Duplicate repositories: {duplicate_repos}")
