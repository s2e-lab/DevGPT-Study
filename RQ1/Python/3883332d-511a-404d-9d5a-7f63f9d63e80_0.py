# Pre-calculate FizzBuzz for the range 1-15
FIZZBUZZ_LOOKUP = [
    "1", "2", "Fizz", "4", "Buzz", "Fizz", "7", "8", "Fizz", "Buzz", "11", "Fizz", "13", "14", "FizzBuzz"
]

FIZZBUZZ_STRING = ''.join(FIZZBUZZ_LOOKUP)

# Since we want to flush efficiently, let's aim for a large buffer size.
# A typical OS page size might be 4096 bytes, but for our purposes and modern hardware, 
# we can use a much larger buffer size. Let's say 1MB for this example.
BUFFER_SIZE = 1 * 1024 * 1024  # 1MB

def generate_large_fizzbuzz():
    repetitions_needed = BUFFER_SIZE // len(FIZZBUZZ_STRING)
    return FIZZBUZZ_STRING * repetitions_needed

if __name__ == "__main__":
    large_fizzbuzz = generate_large_fizzbuzz()

    # For the purpose of the example, we'll just print out a certain number of these large chunks.
    # In practice, you'd write this to your desired output (disk, network, etc.).
    for _ in range(100):  # printing 100MB of FizzBuzz for demonstration purposes
        print(large_fizzbuzz, end="")
