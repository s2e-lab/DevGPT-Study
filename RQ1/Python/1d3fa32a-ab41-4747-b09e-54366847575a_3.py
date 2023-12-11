from bs4 import BeautifulSoup

html = '''
<!DOCTYPE html>
<html>
<head>
    <title>My Web Page</title>
</head>
<body>
    <h1>Welcome to my website</h1>
    <p>This is a paragraph.</p>
    <ul>
        <li>Item 1</li>
        <li>Item 2</li>
    </ul>
    <a href="https://www.example.com">Click here</a>
</body>
</html>
'''

# Parse the HTML content
soup = BeautifulSoup(html, 'html.parser')

# Find the first <h1> tag
h1 = soup.find('h1')
print(h1.text)

# Find all <li> tags
li_tags = soup.find_all('li')
for li in li_tags:
    print(li.text)

# Find an element by attribute
a_tag = soup.find('a', href='https://www.example.com')
print(a_tag.text)
