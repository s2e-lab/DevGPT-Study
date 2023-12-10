from typing import List

def isAcronym(words: List[str], s: str) -> bool:
    # Check if the number of words in the list matches the length of the acronym
    if len(words) != len(s):
        return False
    
    # Check if the first letter of each word in the list matches the corresponding letter in the acronym
    for i in range(len(words)):
        if words[i][0] != s[i]:
            return False
    
    # If all checks pass, return True
    return True
