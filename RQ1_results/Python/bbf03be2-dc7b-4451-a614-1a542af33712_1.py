from googletrans import Translator
from translation_cache import cache_translation

@cache_translation
def translate_text(text: str, target: str = "en") -> str:
    translator = Translator()
    translation = translator.translate(text, dest=target)
    return translation.text

# Test the function
if __name__ == "__main__":
    print(translate_text("Hello", target="es"))
