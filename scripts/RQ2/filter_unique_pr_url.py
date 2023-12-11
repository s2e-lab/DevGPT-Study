def extract_unique_urls(input_file, output_file):
    with open(input_file, 'r') as file:
        urls = file.readlines()

    unique_urls = sorted(set(urls))

    with open(output_file, 'w') as file:
        for url in unique_urls:
            file.write(url)


input_file = 'pr_url_java.txt'
output_file = 'pr_unique_url_java.txt'

extract_unique_urls(input_file, output_file)
