# Assume the necessary imports and initialization here
model = ContextAwareFileSplittingModel(...)
splitter = SplittingAI(model)
document = load_document(...)  # Load your document here
split_points = splitter.propose_split_points(document)
