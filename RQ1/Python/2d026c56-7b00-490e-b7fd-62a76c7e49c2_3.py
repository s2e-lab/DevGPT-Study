def tune_getter(path, param, spec_val):
    if spec_val == LORA_FREEZE:
        return EmptyNode
    if spec_val == LORA_FULL:
        return param

    if len(param.shape) == 1:
        raise ValueError(f'Vectors must either be frozen or fully tuned, but got spec value {spec} for param with path {path}')
    if len(param.shape) == 2:
        b_dim, a_dim = param.shape

        print(f'b_dim: {b_dim}, a_dim: {a_dim}, spec_val: {spec_val}')
        b = jnp.zeros((b_dim, spec_val), dtype=param.dtype)
        a = jax.random.normal(rng, (spec_val, a_dim), dtype=param.dtype) * stddev
        return LoraNode(a, b, alpha=alpha)

    # conv case
