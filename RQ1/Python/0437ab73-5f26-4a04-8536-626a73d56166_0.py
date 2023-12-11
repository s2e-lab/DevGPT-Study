from transliterate import translit

def convert_to_nepali(string):
    nepali_text = translit(string, 'ne', reversed=True)
    return nepali_text

# Example usage
english_text = "Hello, how are you?"
nepali_text = convert_to_nepali(english_text)
print(nepali_text)
