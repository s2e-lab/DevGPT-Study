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

# Get the contents of the <h1> tag
title_text = soup.body.div.h1.get_text()
print(title_text)  # This is a title

# Get the contents of the <p> tag
p_text = soup.body.div.p.get_text()
print(p_text)  # This is a paragraph
