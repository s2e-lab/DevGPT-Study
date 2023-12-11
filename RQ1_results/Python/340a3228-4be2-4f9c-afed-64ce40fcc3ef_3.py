def translate_to_elvish(word):
    elvish_dict = {
        "friend": "mellon",
        "water": "ailin"
    }
    return elvish_dict.get(word, "Unknown word in Elvish")

if "--translate-elvish" in sys.argv:
    index = sys.argv.index("--translate-elvish") + 1
    if index < len(sys.argv):
        word = sys.argv[index]
        print(f"'{word}' in Elvish is: {translate_to_elvish(word)}")
    else:
        print("Please provide a word to translate.")
