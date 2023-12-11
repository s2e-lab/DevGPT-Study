from urllib.parse import quote

author_id = URIRef(quote(f'https://gams.uni-graz.at/o:szd.bibliothek#{author_name[1].strip()}_{author_name[0].strip()}'))
book_id = URIRef(quote(f'https://gams.uni-graz.at/o:szd.bibliothek#{row["ID"]}'))
