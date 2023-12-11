if len(author_name) == 2:
    g.add((author_id, schema.givenName, Literal(author_name[1].strip())))
    g.add((author_id, schema.familyName, Literal(author_name[0].strip())))
