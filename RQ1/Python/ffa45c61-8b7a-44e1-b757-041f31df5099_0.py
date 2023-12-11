import requests
import pandas as pd
import matplotlib.pyplot as plt

# Fetch GDP data
gdp_url = "http://api.worldbank.org/v2/country/all/indicator/NY.GDP.MKTP.CD?date=2020&format=json&per_page=300"
gdp_response = requests.get(gdp_url)
gdp_data = gdp_response.json()[1]

# Fetch area data
area_url = "http://api.worldbank.org/v2/country/all/indicator/AG.SRF.TOTL.K2?date=2020&format=json&per_page=300"
area_response = requests.get(area_url)
area_data = area_response.json()[1]

# Extract and clean the data
gdp_dict = {item['country']['id']: item['value'] for item in gdp_data if item['value']}
area_dict = {item['country']['id']: item['value'] for item in area_data if item['value']}

# Create a list of dictionaries, each containing the data for one country
data = [
    {"CountryCode": country_code, "GDP": gdp, "Area": area_dict[country_code]}
    for country_code, gdp in gdp_dict.items()
    if country_code in area_dict
]

# Create a DataFrame
df = pd.DataFrame(data)

# Plot the data
plt.figure(figsize=(10, 5))
plt.scatter(df['GDP'], df['Area'], alpha=0.5)
plt.title('Countries by GDP and Area (2020)')
plt.xlabel('GDP (Current USD)')
plt.ylabel('Area (sq. km)')
plt.xscale('log')
plt.yscale('log')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.show()
