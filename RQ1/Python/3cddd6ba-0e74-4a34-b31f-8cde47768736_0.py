def sort_by_reversed_strings(numbers):
    # Convert the numbers to strings and reverse those strings
    str_nums = [str(n)[::-1] for n in numbers]

    # Sort based on the reversed strings
    sorted_indexes = sorted(range(len(str_nums)), key=lambda i: str_nums[i])

    # Reorder the original numbers based on the sorted indexes
    return [numbers[i] for i in sorted_indexes]
