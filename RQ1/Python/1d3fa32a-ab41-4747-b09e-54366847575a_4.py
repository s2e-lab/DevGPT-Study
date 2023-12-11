from bs4 import BeautifulSoup

html = '''
<html>
<head>
    <title>Sample Page</title>
</head>
<body>
    <div class="container">
        <h1>This is a title</h1>
        <p>This is a paragraph.</p>
        <a href="https://www.example.com">Click here</a>
    </div>
</body>
</html>
'''

soup = BeautifulSoup(html, 'html.parser')

# Extract text
title_text = soup.title.text
print(title_text)  # This is a title

# Extract attribute
link_href = soup.a['href']
print(link_href)  # https://www.example.com
