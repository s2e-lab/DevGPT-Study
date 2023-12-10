# Define input size
input_size = len(vectorizer.get_feature_names_out())

# Check that targets match data
assert (np.array(train) == np.array(dataset.data[cutoff:])).all()
assert (np.array(test) == np.array(dataset.data[:cutoff])).all()

# Create Dataloaders
train_dataloader = DataLoader(torch_train, batch_size=batch_size, shuffle=True, drop_last=True)
test_dataloader = DataLoader(torch_test, batch_size=batch_size, shuffle=True, drop_last=True)
