from googletrans import Translator

def translate_to_french(text):
    translator = Translator(service_urls=['translate.google.com'])
    translation = translator.translate(text, dest='fr')
    return translation.text

# Example usage
input_text = "Hello, how are you?"
translated_text = translate_to_french(input_text)
print(translated_text)
