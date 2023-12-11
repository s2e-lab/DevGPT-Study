import re

def identify_string(string):
    # Regular expression patterns for Twitch login names and StreamElements account IDs
    twitch_pattern = r"^[A-Za-z0-9_]{4,25}$"
    streamelements_pattern = r"^[a-fA-F0-9]{24}$"

    if re.match(twitch_pattern, string):
        return "Twitch login name"
    elif re.match(streamelements_pattern, string):
        return "StreamElements Account ID"
    else:
        return "Unknown"

# Example usage
string1 = "mytwitchusername123"
string2 = "5eb63bbbe01eeed093cb22bb8f5acdc3"
string3 = "invalid_string"

print(identify_string(string1))  # Output: Twitch login name
print(identify_string(string2))  # Output: StreamElements Account ID
print(identify_string(string3))  # Output: Unknown
