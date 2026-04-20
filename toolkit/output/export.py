import csv
import json
from pathlib import Path

from toolkit.utils import normalize_row as _normalize_item


def to_json(results, path):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    normalized = [_normalize_item(item) for item in results] if isinstance(results, list) else _normalize_item(results)
    target.write_text(json.dumps(normalized, indent=2, ensure_ascii=False), encoding="utf-8")
    return target


def to_csv(results, path):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = [_normalize_item(item) for item in results]
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return target


