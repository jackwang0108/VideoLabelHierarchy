# Standard Library
import json
from pathlib import Path
from typing import Any


def load_json(file_path: Path) -> Any:
    with file_path.open(mode="r") as f:
        return json.load(f)
