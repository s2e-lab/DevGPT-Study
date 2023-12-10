from geopy.geocoders import Nominatim
from geopy import geocoders

geocoders.options.default_user_agent = "my-custom-user-agent"

geolocator = Nominatim()
