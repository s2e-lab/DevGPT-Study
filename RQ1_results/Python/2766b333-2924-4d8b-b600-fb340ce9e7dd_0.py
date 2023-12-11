from sklearn.feature_extraction.text import CountVectorizer

# Create a CountVectorizer object
vectorizer = CountVectorizer()

# Define an array of strings
array_of_strings = ["Hello world", "This is an example"]

# Transform the array of strings
transformed_data = vectorizer.transform(array_of_strings)

# Print the transformed data
print(transformed_data)
