from geopy.point import Point

def normalize_coordinates(coordinates_str):
    # Entferne Leerzeichen und ersetze Kommas durch Punkte
    coordinates_str = coordinates_str.replace(" ", "").replace(",", ".")
    # Parsen und normalisieren der Koordinaten
    point = Point(coordinates_str)
    # Holen Sie sich die normalisierten Koordinaten im Dezimalformat
    latitude = point.latitude
    longitude = point.longitude
    return latitude, longitude

# Beispiel-Koordinaten
coordinates_str = "48°15'48,350\"N 11°18'42,930\"E"

# Koordinaten normalisieren
latitude, longitude = normalize_coordinates(coordinates_str)
print("Normalisierte Koordinaten:")
print("Breitengrad:", latitude)
print("Längengrad:", longitude)
