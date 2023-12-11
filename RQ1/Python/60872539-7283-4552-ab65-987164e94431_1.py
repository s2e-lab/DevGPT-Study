import requests

api_endpoint = "https://api.weatherprovider.com/current"
location = "NewYork"

# Send a GET request to the API endpoint with the location parameter
response = requests.get(api_endpoint, params={"location": location})

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Extract the weather data from the response
    weather_data = response.json()
    # Process and display the weather information
    print("Current temperature in New York:", weather_data["temperature"])
    print("Weather description:", weather_data["description"])
else:
    print("Failed to retrieve weather data")
