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
        <p>This is another paragraph.</p>
        <a href="https://www.example.com">Click here</a>
    </div>
</body>
</html>
'''

soup = BeautifulSoup(html, 'html.parser')

# Find all <p> tags
p_tags = soup.find_all('p')
for p in p
