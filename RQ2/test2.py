def count_unique_urls(file_path):
    with open(file_path, 'r') as file:
        urls = file.read().splitlines()
        unique_urls = set(urls)
        return len(unique_urls)

file_path = '../../results/pr_urls.txt'
unique_count = count_unique_urls(file_path)
print(f"Number of unique URLs: {unique_count}")
