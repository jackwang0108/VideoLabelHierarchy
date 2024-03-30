# Standard Library
from pathlib import Path
from functools import lru_cache
from typing import Literal, Tuple, get_args

# Third-Party Library
import pandas as pd

# My Library
from .color import red


CSVFields = Tuple[Literal["name", "yt_id", "height", "width", "fps"]]


@lru_cache(maxsize=None)
def parse_csv(csv_path: Path | str, fields: CSVFields) -> pd.DataFrame:
    existing_fields = get_args(get_args(CSVFields)[0])
    assert all(
        (i in existing_fields) for i in fields
    ), f"all fileds should be in {existing_fields}, got {red(fields, True)}"

    def process_fps(fps: float) -> float:
        return round(fps, ndigits=3)

    df = pd.read_csv(csv_path)
    df["fps"] = df["fps"].apply(process_fps)
    df[["width", "height"]] = df["resolution"].str.split("x", expand=True)

    column_order = fields
    df = df.reindex(columns=column_order)
    return df


if __name__ == "__main__":
    print()
