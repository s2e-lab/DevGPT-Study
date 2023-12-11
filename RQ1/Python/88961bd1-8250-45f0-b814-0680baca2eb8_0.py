import hamilton as ht

# Define the pipeline
pipeline = ht.Pipeline()

# Node to get the user message and prepare it for the GPT-4 model
@pipeline.node()
def prepare_message(user_message):
    return f"{user_message}. Reply in emojis"

# Node to get the emoji response from the GPT-4 model
@pipeline.node(depends_on='prepare_message')
def get_emoji_response(message, gpt4_model):
    # Assuming a function `query_gpt4` that communicates with the GPT-4 model
    emoji_response = query_gpt4(message, model=gpt4_model, temperature=0)
    return emoji_response

# Node to prepare the emoji response for translation
@pipeline.node(depends_on='get_emoji_response')
def prepare_translation(emoji_response):
    return f"Translate this emoji message {emoji_response} to plain english"

# Node to get the translated response from the GPT-4 model
@pipeline.node(depends_on='prepare_translation')
def get_translation(message, gpt4_model):
    translated_response = query_gpt4(message, model=gpt4_model)
    return translated_response

# Running the pipeline
result = pipeline.run({'user_message': "Hey there, how is it going?", 'gpt4_model': 'gpt-4'})
print(result['get_translation'])
