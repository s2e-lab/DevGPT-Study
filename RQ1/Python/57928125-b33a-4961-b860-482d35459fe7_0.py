import re

strings = [
    "Hello, check out this website: https://example.com",
    "I found an interesting article at https://example.org",
    "Click here: http://example.net for more information"
]

pattern = r"https?://\S+"

cleaned_strings = []

for string in strings:
    cleaned_string = re.sub(pattern, "", string)
    cleaned_strings.append(cleaned_string)

print(cleaned_strings)
