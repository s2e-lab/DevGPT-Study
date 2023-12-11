models = ["text-davinci-003", "text-davinci-001", "davinci-instruct-beta", "davinci"]
prompt = "[System: You are a helpful assistant]\n\nUser: Write a unique, surprising, extremely randomized story with highly unpredictable changes of events.\n\nAI:"

for model in models:
    sequences = set()
    for _ in range(30):
        completion = openai.Completion.create(
            model=model,
            prompt=prompt,
            max_tokens=256,
            temperature=0
        )
        sequences.add(completion.choices[0].text)
    print(f"Model {model} created {len(sequences)} unique sequences.")
