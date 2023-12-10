def analyze_repos_and_urls(file_path):
    repo_counts = {}
    unique_urls = set()

    with open(file_path, 'r') as file:
        for line in file:
            url = line.strip()
            unique_urls.add(url)  # Add URL to the set for unique counting

            parts = url.split('/')
            # Extract just the repository name
            repo_name = parts[4] if len(parts) > 4 else None
            if repo_name:
                repo_counts[repo_name] = repo_counts.get(repo_name, 0) + 1

    # Count of all distinct URLs
    total_unique_urls = len(unique_urls)

    # Count of unique repositories
    unique_repo_count = len(repo_counts)

    duplicates = {repo: count for repo,
                  count in repo_counts.items() if count > 1}

    return unique_repo_count, duplicates, total_unique_urls


file_path = 'pr_unique_url_python.txt'  # Replace with your file path
unique_repo_count, duplicate_repos, total_unique_urls = analyze_repos_and_urls(
    file_path)

print(f"Number of unique repositories: {unique_repo_count}")    # 39
print("Duplicate repositories and their counts:")
for repo, count in duplicate_repos.items():
    print(f"{repo}: {count} times")
print(f"Total number of different URLs: {total_unique_urls}")
