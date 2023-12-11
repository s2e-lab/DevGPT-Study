entity = FunctionEntity(
    name="my_function",
    docstring="This is a function.",
    signature="(arg1, arg2)",
    embedding_model=MyEmbeddingModel(),
    embedding_storage=WeaviateEmbeddingStorage()
)
