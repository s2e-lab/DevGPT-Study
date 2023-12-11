net = MomentNetwork(
    ...,
    optimizer_cls=torch.optim.Adam,
    optimizer_args={"lr": 1e-4, "weight_decay": 1e-5}
)
