import math

def calculate_bounds(k, X):
    term1 = (2 * math.pi * k) ** (1 / (2 * k))
    term2 = (k / math.e) ** (k / k)
    term3_lower = math.exp(1 / (12 * k + 1) / k)
    term3_upper = math.exp(1 / (12 * k) / k)
    
    lower_bound = term1 * term2 * term3_lower
    upper_bound = term1 * term2 * term3_upper
    
    return lower_bound, upper_bound

# Example usage:
k = 5
X = 1.5
lower_bound, upper_bound = calculate_bounds(k, X)
print(f"For k = {k} and X = {X}:")
print(f"Lower Bound (n >=): {lower_bound}")
print(f"Upper Bound (n <=): {upper_bound}")
