import random

def biased_multiplier():
    x = random.uniform(1, 10)  # Generate a random number between 1 and 10
    probability = 1 / x        # Calculate the probability inversely proportional to x
    random_value = random.uniform(0, 1)
    
    if random_value < probability:
        return x
    else:
        return biased_multiplier()  # Recursively call the function until a value is selected

# Example usage
result = biased_multiplier()
print(result)
