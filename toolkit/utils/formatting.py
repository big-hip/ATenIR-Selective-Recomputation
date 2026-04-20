from dataclasses import asdict, is_dataclass


def format_bytes(n: int) -> str:
    value = float(n)
    for unit in ["B", "KB", "MB", "GB"]:
        if abs(value) < 1024:
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


def normalize_row(row):
    if is_dataclass(row):
        return asdict(row)
    if isinstance(row, dict):
        return row
    return vars(row)
