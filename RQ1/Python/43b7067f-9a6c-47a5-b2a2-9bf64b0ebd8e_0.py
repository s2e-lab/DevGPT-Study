class MomentNetwork:
    """Class for the Moment Network."""

    def __init__(
        self,
        net: nn.Module,
        model: Callable,
        search_space: SearchSpace,
        N: int,
        sampler: BaseSampler,
        criterion: nn.Module = MSE_LOSS,
        optimizer_cls: Optional[Type[torch.optim.Optimizer]] = torch.optim.Adam,
        optimizer_args: Optional[Dict[str, Any]] = None,
        batch_dtype: torch.dtype = torch.float32,
        verbosity: int = 10,
    ):
        """Initialise a Moment Network.

        Args:
            ... [other arguments] ...
            optimizer_cls (Type[torch.optim.Optimizer], optional): Optimizer class to be used. Defaults to torch.optim.Adam.
            optimizer_args (Dict[str, Any], optional): Arguments to be passed to the optimizer during its creation. Defaults to None.
            ... [other arguments] ...
        """
        self.net = net
        self.model = model
        self.criterion = criterion

        # Setting up optimizer
        default_optimizer_args = {"lr": 1e-3}
        if optimizer_args is None:
            optimizer_args = default_optimizer_args
        else:
            optimizer_args = {**default_optimizer_args, **optimizer_args}  # Overwrite default args if provided
        self.optimizer = optimizer_cls(self.net.parameters(), **optimizer_args)

        # Sampler and bounds for model parameters
        self.sampler = sampler
        self.search_space = search_space

        self.N = N
        self.batch_dtype = batch_dtype
        self.verbosity = verbosity
