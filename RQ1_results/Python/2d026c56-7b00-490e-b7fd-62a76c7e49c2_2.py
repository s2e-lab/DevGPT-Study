def freeze_getter(param, spec_val):
    if spec_val == LORA_FULL:
        return EmptyNode
    return param
