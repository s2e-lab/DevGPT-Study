import ebooklib
from ebooklib import epub

book = epub.read_epub('path_to_your_book.epub')

for image in book.get_items_of_type(ebooklib.ITEM_IMAGE):
    print(image)
