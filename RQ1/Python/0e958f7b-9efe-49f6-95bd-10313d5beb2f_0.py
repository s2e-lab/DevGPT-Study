def arabic_to_roman(number):
    if not isinstance(number, int) or number <= 0 or number >= 4000:
        raise ValueError("Invalid input: Only positive integers between 1 and 3999 are supported.")

    roman_numerals = [
        ("M", 1000),
        ("CM", 900),
        ("D", 500),
        ("CD", 400),
        ("C", 100),
        ("XC", 90),
        ("L", 50),
        ("XL", 40),
        ("X", 10),
        ("IX", 9),
        ("V", 5),
        ("IV", 4),
        ("I", 1)
    ]

    roman_numeral = ""
    for numeral, value in roman_numerals:
        while number >= value:
            roman_numeral += numeral
            number -= value

    return roman_numeral

# Test the function
try:
    number = int(input("Enter an Arabic number (1 to 3999): "))
    roman_numeral = arabic_to_roman(number)
    print(f"{number} in Roman numerals is: {roman_numeral}")
except ValueError as e:
    print(e)
