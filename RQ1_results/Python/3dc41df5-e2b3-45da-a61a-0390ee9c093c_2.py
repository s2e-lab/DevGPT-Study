artist_name = 'Taylor Swift'
results = sp.search(q=artist_name, type='artist')

if results['artists']['items']:
    artist = results['artists']['items'][0]
    print("Artist Name:", artist['name'])
    print("Followers:", artist['followers']['total'])
    print("Popularity:", artist['popularity'])
else:
    print("Artist not found.")
