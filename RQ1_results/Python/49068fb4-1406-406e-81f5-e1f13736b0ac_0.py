from transformers import TrainingArguments

training_args = TrainingArguments(
    # Other arguments...
    learning_rate=5e-5,  # Typically a small learning rate is used for fine-tuning
    lr_scheduler_type="linear",
    warmup_ratio=0.1,  # The proportion of total training steps used for warm-up
    # More arguments...
)
