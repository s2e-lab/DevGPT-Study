from bs4 import BeautifulSoup
import re

def extract_emails_from_html(file_path):
    with open(file_path, 'r') as f:
        data = f.read()

    soup = BeautifulSoup(data, 'html.parser')

    # Regular expression to match most email addresses
    email_regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'

    emails = re.findall(email_regex, str(soup))

    return emails
