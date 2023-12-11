class TwentyNewsgroupsDataset(Dataset):
    def __init__(self,texts,targets, vectorizer):
        self.targets = targets
        self.data = texts # converts the sparse matrix to a dense one... I think.
        self.vectorizer = vectorizer
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        vector = torch.tensor(self.vectorizer.transform([self.data[idx]]).todense()) # torch.tensor() to convert the matrix into a torch tensor for training
        label = self.targets[idx]
        return vector, label
