# Standard Library
from pathlib import Path

# Third-Party Library

# My Library
from .io import load_yaml

# TODO: 增加argparser


def parse_config_yaml(yaml_path: Path) -> dict[str, dict[str, str]]:
    """
    Parses a YAML configuration file, replacing variables with their corresponding values defined in YAML files.

    Args:
        yaml_path: Path to the YAML configuration file.

    Returns:
        A dictionary representing the parsed configuration with variables replaced.
    """
    raw_content = load_yaml(yaml_path)

    def replace_variables(unit: str | dict[str, str] | list[dict[str, str]], var_name: str, var_value: str):
        """ depth-first traversal """

        if isinstance(unit, str):
            return unit.replace(f"${{{var_name}}}", var_value)
        elif isinstance(unit, (list, dict)):
            for k in enumerate(unit) if isinstance(unit, list) else unit.keys():
                unit[k] = replace_variables(unit[k], var_name, var_value)

        return unit

    for var_name, var_value in raw_content["variables"].items():
        raw_content = replace_variables(raw_content, var_name, var_value)
    return raw_content


if __name__ == "__main__":
    import pprint
    pprint.pprint(
        parse_config_yaml(Path(__file__).parent.joinpath("../config/vlh.yaml")))
