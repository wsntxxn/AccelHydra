import json
from typing import Union, Dict
from pathlib import Path
import os

from pathlib import Path


def read_jsonl_to_mapping(
    jsonl_file: Union[str, Path],
    key_col: str,
    value_col: str,
    base_path=None
) -> Dict[str, str]:
    """
    Read two columns, indicated by `key_col` and `value_col`, from the
    given jsonl file to return the mapping dict
    TODO handle duplicate keys
    """
    mapping = {}
    with open(jsonl_file, 'r') as file:
        for line in file.readlines():
            data = json.loads(line.strip())
            key = data[key_col]
            value = data[value_col]
            if base_path:
                value = os.path.join(base_path, value)
            mapping[key] = value
    return mapping
