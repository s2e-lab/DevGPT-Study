import re

text = "The values are 85%, 90%, 79.5%, and 100%. The range should be between 80% and 90%."
matches = re.findall(regex_pattern, text)
print(matches)
