def next(self, messages: list[dict[str, str]], prompt=None):
    if prompt:
        messages += [{"role": "user", "content": prompt}]

    logger.debug(f"Creating a new chat completion: {messages}")

    # Retry the request up to 3 times
    for _ in range(3):
        try:
            response = openai.ChatCompletion.create(
                messages=messages,
                stream=True,
                model=self.model,
                temperature=self.temperature,
            )
            break
        except requests.exceptions.ChunkedEncodingError:
            logger.warning("ChunkedEncodingError occurred, retrying the request...")
    else:
        raise Exception("Failed to create chat completion after 3 attempts")

    chat = []
    for chunk in response:
        delta = chunk["choices"][0]["delta"]
        msg = delta.get("content", "")
        print(msg, end="")
        chat.append(msg)
    print()
    messages += [{"role": "assistant", "content": "".join(chat)}]
    logger.debug(f"Chat completion finished: {messages}")
    return messages
