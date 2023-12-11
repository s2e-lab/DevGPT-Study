import re

def clean_transcript(text):
    # Merge the split lines
    split_text = re.split(r'(?<=\.|- )\n', text)

    # Join the lines back together
    joined_text = ' '.join([line.replace('\n', ' ') for line in split_text])

    # Remove speaker indications
    clean_text = re.sub(r'- ', '', joined_text)

    return clean_text

# Your YouTube transcript
transcript = """
- It's not our business to
change the Russian government. And anybody who thinks it's a
good idea to do regime change, in Russia, which has more
nuclear weapons than we do, is, I think, irresponsible. And, you know, Vladimir
Putin himself has had, you know, we will not live
in a world without Russia. And it was clear when he said that, that he 

was talking about himself, and he has his hand on a button
that could bring, you know, Armageddon to the entire planet. So why are we messing with this? It's not our job to change that regime
