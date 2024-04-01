# Standard Library
import json
from pathlib import Path
from typing import Any

# Third-Party Library
import yaml


def load_json(file_path: Path) -> Any:
    with file_path.open(mode="r") as f:
        return json.load(f)


def load_yaml(file_path: Path) -> dict[str, dict[str, str]]:
    with file_path.open(mode="r") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    import pprint
    pprint.pprint(
        load_yaml(Path(__file__).parent.joinpath("../config/vlh.yaml")))
