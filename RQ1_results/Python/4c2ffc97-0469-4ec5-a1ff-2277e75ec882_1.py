import requests

def is_valid_link(url):
    try:
        response = requests.get(url)
        # Check if the response status code is in the 2xx range (successful)
        return response.status_code // 100 == 2
    except requests.exceptions.RequestException:
        # Exception occurred, link is not valid
        return False

# Example URL to validate
url = "https://www.example.com"

if is_valid_link(url):
    print("The link is valid and accessible.")
else:
    print("The link is invalid or not accessible.")
