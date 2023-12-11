import sqlite3

# Open connection to the new database
new_db = sqlite3.connect('favorites.db')

# Attach the old database to the new database
new_db.execute("ATTACH DATABASE 'favorites old.db' AS old_db")

# Insert unique data from the old database to the new database
new_db.execute("""
    INSERT INTO favorites (name, url, mode, image, duration, quality)
    SELECT name, url, mode, image, duration, quality FROM old_db.favorites
    WHERE (name, url, mode, image, duration, quality) NOT IN (SELECT name, url, mode, image, duration, quality FROM favorites)
""")

# Commit changes and close the connection
new_db.commit()
new_db.close()
