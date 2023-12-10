import random
import re
import requests

def find_document_source(unique_phrases, results):
    high_probability_source = get_high_probability_source(results)
    
    if high_probability_source['is_high']:
        print("High Probability Source:")
        print("Title:", high_probability_source['result']['title'])
        print("Link:", high_probability_source['result']['link'])
        print("Description:", high_probability_source['result']['snippet'])
    elif high_probability_source['is_good']:
        print("Good Probability Source:")
        print("Title:", high_probability_source['result']['title'])
        print("Link:", high_probability_source['result']['link'])
        print("Description:", high_probability_source['result']['snippet'])
    else:
        print("No definitive source identified.")

def get_high_probability_source(results):
    link_frequency = {}
    for result in results:
        link = result['link']
        link_frequency[link] = link_frequency.get(link, 0) + 1
    
    high_frequency_link = max(link_frequency, key=link_frequency.get)
    high_frequency_count = link_frequency[high_frequency_link]
    
    if high_frequency_count == 3:
        return {'result': next(result for result in results if result['link'] == high_frequency_link), 'is_high': True, 'is_good': False}
    elif high_frequency_count == 2:
        return {'result': next(result for result in results if result['link'] == high_frequency_link), 'is_high': False, 'is_good': True}
    else:
        return {'is_high': False, 'is_good': False}

def generate_unique_phrases(document, num_phrases=3, phrase_length=8):
    words = re.findall(r'\w+', document)
    unique_phrases = set()
    while len(unique_phrases) < num_phrases:
        start_index = random.randint(0, len(words) - phrase_length)
        phrase = ' '.join(words[start_index:start_index + phrase_length])
        unique_phrases.add(phrase)
    return list(unique_phrases)

if __name__ == "__main__":
    document = """
    This is the content of your document. Replace this with your actual document content.
    """
    unique_phrases = generate_unique_phrases(document)
    
    # Replace 'YOUR_SEARCH_ENGINE_API_KEY' and 'YOUR_SEARCH_ENGINE_CUSTOM_SEARCH_ID' with actual values
    search_engine_api_key = 'YOUR_SEARCH_ENGINE_API_KEY'
    search_engine_cx = 'YOUR_SEARCH_ENGINE_CUSTOM_SEARCH_ID'
    base_url = f'https://www.googleapis.com/customsearch/v1'
    
    for phrase in unique_phrases:
        query_params = {
            'key': search_engine_api_key,
            'cx': search_engine_cx,
            'q': phrase
        }
        response = requests.get(base_url, params=query_params)
        results = response.json().get('items', [])
        
        print("Search phrase:", phrase)
        
        for result in results:
            print("Title:", result['title'])
            print("Link:", result['link'])
            print("Description:", result['snippet'])
            print("="*50)
            
        find_document_source(unique_phrases, results)
