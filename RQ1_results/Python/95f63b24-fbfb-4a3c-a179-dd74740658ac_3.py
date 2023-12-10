input_text = "Hello, Llama!"
input_ids = tokenizer.encode(input_text, return_tensors="pt")
output = model.generate(input_ids)
decoded_output = tokenizer.decode(output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
print(decoded_output)
