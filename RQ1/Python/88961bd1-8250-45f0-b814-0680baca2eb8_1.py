import asyncio
import hamilton as ht

# Define the pipeline
pipeline = ht.Pipeline()

# Node to generate a range of numbers
@pipeline.node()
def generate_numbers(n):
    return list(range(n))

# Node to apply a delay and return a formatted string for each number
@pipeline.node(depends_on='generate_numbers', apply_async=True)
async def format_numbers(numbers):
    formatted_numbers = []
    for number in numbers:
        await asyncio.sleep(1)
        formatted_numbers.append(f"Number: {number}")
    return formatted_numbers

# Running the pipeline
result = pipeline.run({'n': 3})
print(result['format_numbers'])
