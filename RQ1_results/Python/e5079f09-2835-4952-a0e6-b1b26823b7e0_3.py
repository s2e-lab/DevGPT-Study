from decouple import config

# settings.py

...

# Rasa application URL
RASA_APP_URL = config('RASA_APP_URL', default='http://default.rasa.url.here/')
