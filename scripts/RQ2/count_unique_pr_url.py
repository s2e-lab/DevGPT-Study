def count_unique_urls(file_path):
    with open(file_path, 'r') as file:
        urls = file.readlines()
        unique_urls = set(url.strip() for url in urls)
        return len(unique_urls)


file_path = 'pr_urls.txt'
unique_url_count = count_unique_urls(file_path)
print(f"Number of unique URLs: {unique_url_count}")
# Note: Number of unique URLs: 44
