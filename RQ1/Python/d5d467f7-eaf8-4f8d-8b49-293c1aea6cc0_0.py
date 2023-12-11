from torch.utils.data import random_split

# Let's assume `dataset` is your PyTorch Dataset object
train_size = int(0.8 * len(dataset))  # 80% for training
val_size = len(dataset) - train_size  # rest for validation

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
