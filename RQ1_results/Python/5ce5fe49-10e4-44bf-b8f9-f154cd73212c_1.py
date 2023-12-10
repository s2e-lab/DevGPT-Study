import requests
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Define the URL for the data file
url = "https://berkeley-earth-temperature-hr.s3.amazonaws.com/Global_TAVG_annual.txt"

# Function to skip lines starting with '%'
def skip_comments(file):
    for line in file:
        if not line.startswith('%'):
            yield line

# Initialize empty lists for years, temperatures, and errors
years = []
temperatures = []
errors = []

# Fetch data from the URL
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    # Decode the content and split it into lines
    lines = response.text.splitlines()

    # Iterate through non-comment lines and extract data
    for line in skip_comments(lines):
        data = line.split()  # Split the line into columns
        year = int(data[0])  # Year is in the first column
        temperature = float(data[1])  # Temperature is in the second column
        error = float(data[2])  # Error is in the third column

        # Append data to the respective lists
        years.append(year)
        temperatures.append(temperature)
        errors.append(error)

    # Convert lists to numpy arrays for plotting and fitting
    years = np.array(years)
    temperatures = np.array(temperatures)
    errors = np.array(errors)

    # Filter data from 1960 to 2023
    mask = (years >= 1960) & (years <= 2023)
    years_filtered = years[mask]
    temperatures_filtered = temperatures[mask]

    # Define a quadratic function for fitting
    def quadratic_fit(x, a, b, c):
        return a * x**2 + b * x + c

    # Fit the data with the quadratic function
    popt, _ = curve_fit(quadratic_fit, years_filtered, temperatures_filtered)

    # Print out the fit parameters
    a, b, c = popt
    print(f"Quadratic Fit Parameters:")
    print(f"a: {a:.4f}")
    print(f'b: {b:.4f}')
    print(f'c: {c:.4f}')

    # Create a plot with error bars and the fitted curve
    plt.figure(figsize=(10, 6))
    plt.errorbar(years, temperatures, yerr=errors, fmt='o-', color='black', label='Temperature (°C)')
    
    # Generate values for the fitted curve
    curve_years = np.linspace(1960, 2023, 100)
    fitted_curve = quadratic_fit(curve_years, *popt)
    
    # Plot the fitted curve
    plt.plot(curve_years, fitted_curve, '--', label='Quadratic Fit', color='red')

    plt.title('Year vs. Temperature with Error Bars and Quadratic Fit')
    plt.xlabel('Year')
    plt.ylabel('Temperature (°C)')
    plt.grid(True)
    plt.legend()
    plt.show()
else:
    print("Failed to fetch data. Status code:", response.status_code)
