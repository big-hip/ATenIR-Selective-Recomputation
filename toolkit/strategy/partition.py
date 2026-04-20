from torch._functorch.partitioners import default_partition, min_cut_rematerialization_partition


_PARTITION_FNS = {
    "default": default_partition,
    "min_cut": min_cut_rematerialization_partition,
}


def get_partition_fn(name: str = "default"):
    try:
        return _PARTITION_FNS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown partition: {name}. Choose from {list(_PARTITION_FNS)}") from exc
