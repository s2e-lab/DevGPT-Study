from bs4 import BeautifulSoup

# Create a BeautifulSoup object
html = "<html><body><p>Hello, World!</p></body></html>"
soup = BeautifulSoup(html, 'html.parser')

# Extract the text from the <p> element
text = soup.p.text
print(text)
