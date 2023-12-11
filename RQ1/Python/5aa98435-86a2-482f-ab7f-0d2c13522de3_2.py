combined_attention_mask = _make_causal_mask(
    input_shape,
    inputs_embeds.dtype,
    device=inputs_embeds.device,
    past_key_values_length=past_key_values_length,
)
