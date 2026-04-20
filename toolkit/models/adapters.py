from transformers import LlamaConfig, MistralConfig


def _get_first_attr(config, names: tuple[str, ...], field_name: str) -> int:
    for attr in names:
        if hasattr(config, attr):
            value = getattr(config, attr)
            if value is not None:
                return value
    raise AttributeError(f"Cannot find {field_name} in {type(config).__name__}")


def get_hidden(config) -> int:
    return _get_first_attr(config, ("hidden_size", "n_embd", "d_model"), "hidden_size")


def get_num_layers(config) -> int:
    return _get_first_attr(config, ("num_hidden_layers", "n_layer", "num_layers"), "num_layers")


def get_num_heads(config) -> int:
    return _get_first_attr(config, ("num_attention_heads", "n_head", "num_heads"), "num_heads")


def get_intermediate(config) -> int:
    for attr in ("intermediate_size", "n_inner", "ffn_dim"):
        if hasattr(config, attr):
            value = getattr(config, attr)
            if value is not None:
                return value
    return 4 * get_hidden(config)


def get_vocab_size(config) -> int:
    return config.vocab_size


def has_position_embedding(config) -> bool:
    return not isinstance(config, (LlamaConfig, MistralConfig))


def get_num_kv_heads(config) -> int:
    if hasattr(config, "num_key_value_heads"):
        value = getattr(config, "num_key_value_heads")
        if value is not None:
            return value
    return get_num_heads(config)
