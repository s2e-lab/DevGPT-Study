def generate_commentary(self, dialogue, history):
    print("Generating commentary...\n")
    random_roll = random.randint(0, 100)
    if random_roll > 85:
        # Handle your "extra deep and extravagant" logic here if you want
    
    response = self.conversation.predict(input=dialogue, history=history)
    
    print("Commentary generated...\n")

    if not response:
        return None

    return response
