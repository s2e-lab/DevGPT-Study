def __init__(self):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    self.llm = ChatOpenAI(temperature=1.2, model="gpt-4", max_tokens=250)
    self.prompt = PromptTemplate(
        input_variables=["history", "input"],
        template=_bio  # Use your character's bio or any other template you want
    )
    self.conversation = ConversationChain(
        prompt=self.prompt,
        llm=self.llm,
        memory=ConversationBufferMemory(ai_prefix=_charName)
    )
