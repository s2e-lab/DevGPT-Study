import pandas as pd
import random

# Function to generate a random cost based on the claim type and year
def generate_claim_cost(claim_type, year):
    if claim_type == 'Dentist':
        base_cost = 100
    elif claim_type == 'Psychiatrist':
        base_cost = 150
    elif claim_type == 'General Physician':
        base_cost = 80
    elif claim_type == 'Physiotherapy':
        base_cost = 120
    else:
        base_cost = 50

    # Adjust cost based on year
    if year == 2021:
        base_cost *= 1.1
    elif year == 2023:
        base_cost *= 0.9

    # Add some random variation
    cost = random.uniform(base_cost - 20, base_cost + 20)
    return round(cost, 2)

# Generating sample data
claim_types = ['Dentist', 'Psychiatrist', 'General Physician', 'Physiotherapy']
years = [2021, 2022, 2023]
people = ['John', 'Jane', 'Michael', 'Emily', 'William']

data = []
for year in years:
    for person in people:
        num_claims = random.randint(1, 5)  # Random number of claims per person per year
        for _ in range(num_claims):
            claim_type = random.choice(claim_types)
            cost = generate_claim_cost(claim_type, year)
            date = pd.to_datetime(f"{random.randint(1, 12)}/{random.randint(1, 28)}/{year}", format='%m/%d/%Y')
            data.append([person, claim_type, date, year, cost])

# Create the DataFrame
columns = ['Person', 'ClaimType', 'Date', 'Year', 'Cost']
df = pd.DataFrame(data, columns=columns)

# Display the DataFrame
print(df)
