from gpt_fn import text_generation

input_text = "Once upon a time"
generated_text = text_generation.generate_text(input_text, max_length=100)

print(generated_text)
