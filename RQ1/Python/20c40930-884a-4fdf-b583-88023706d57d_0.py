import torch

# Create a sparse matrix
indices = torch.tensor([[0, 1, 1],
                        [2, 0, 1]])
values = torch.tensor([1, 2, 3], dtype=torch.float32)
sparse_tensor = torch.sparse_coo_tensor(indices, values, size=(3, 3))

# Convert the sparse tensor to a dense tensor
dense_tensor = sparse_tensor.to_dense()
