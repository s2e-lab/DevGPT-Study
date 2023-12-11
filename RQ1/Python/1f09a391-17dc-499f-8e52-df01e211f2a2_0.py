def generate_rap_line(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a rapper composer that uses the bip39 wordlist to rhyme."},
            {"role": "user", "content": prompt}
        ]
    )

    rap_line = response['choices'][0]['message']['content']
    # Find a rhyme from the bip39 wordlist
    rhyme = find_rhyme(rap_line, bip39_wordlist)
    rap_line += " " + rhyme

    return rap_line
