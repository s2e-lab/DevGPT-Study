import PySimpleGUI as sg
from translate import translate_text, LANGUAGES
from translation_cache import Translation, create_db_and_tables, get_translation, add_translation
from sqlmodel import Session

def create_window():
    # ... (same as before)

def main():
    engine = create_db_and_tables()
    session = Session(engine)
    
    window = create_window()
    translations = []

    while True:
        event, values = window.read()

        if event == sg.WINDOW_CLOSED or event == "Exit":
            break

        if event == "Translate" or event == '\r':
            target_language_key = {v: k for k, v in LANGUAGES.items()}[values["-LANG-"]]
            original_text = values["-TEXT-"]
            
            # First, try to get the translation from the cache
            cached_translation = get_translation(original_text, target_language_key, session)
            
            if cached_translation:
                translated_text = cached_translation.translated_text
            else:
                translated_text = translate_text(original_text, target=target_language_key)
                add_translation(original_text, translated_text, target_language_key, session)
                
            # ... (same as before)

if __name__ == "__main__":
    main()
