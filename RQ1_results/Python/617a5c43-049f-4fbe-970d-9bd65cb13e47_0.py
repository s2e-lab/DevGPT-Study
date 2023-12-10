import random

def generate_random_numbers():
    random_numbers = random.sample(range(1, 66), 5)
    return random_numbers

if __name__ == "__main__":
    random_numbers = generate_random_numbers()
    print("Random numbers:", random_numbers)
