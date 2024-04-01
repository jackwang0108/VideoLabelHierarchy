# Standard Library
import copy
import argparse
import contextlib
from pathlib import Path
from ast import literal_eval
from typing import Optional, TypedDict, Literal

# Third-Party Library

# My Library
from .io import load_yaml

# TODO: 增加argparser


class ConfigNode(dict):
    """ Configuration in Tree-like Structure """

    def __init__(
        self,
        key_list: Optional[dict] = None,
        init_dict: Optional[dict] = None,
    ) -> None:
        key_list = [] if key_list is None else key_list
        init_dict = {} if init_dict is None else init_dict
        for key, value in init_dict.items():
            if isinstance(value, dict):
                init_dict[key] = ConfigNode(
                    key_list=key_list + [key], init_dict=value)
        super(ConfigNode, self).__init__(init_dict)

    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError(name)

    def __setattr__(self, name, value) -> None:
        self[name] = value

    def __str__(self) -> str:
        def add_indent(s: str, num_spaces: int):
            s_list: list[str] = s.split("\n")
            if len(s_list) == 1:
                return s
            first = s_list.pop(0)

            # add indent
            s_list = [(num_spaces * " " + line) for line in s_list]

            # merge
            s = "\n".join(s_list)
            s = first + "\n" + s

            return s

        r = ""
        s = []
        # post-order traversal
        for key, value in sorted(self.items()):
            separator = "\n" if isinstance(value, ConfigNode) else " "
            attr_str = f"{key}:{separator}{value}"
            attr_str = add_indent(attr_str, 2)
            s.append(attr_str)
        r += "\n".join(s)
        return r

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({super(ConfigNode, self).__repr__()})"

    def to_yaml(self, yaml_path: Path):
        content: str = str(self)
        with yaml_path.open(mode="w") as f:
            f.write(content)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Command-line tool for training, validating and testing VideoLabelHierarchy to temporal-precisely spot events in videos")

    parser.add_argument(
        "-m", "--mode",
        type=str,
        default="train",
        choices=["train", "test", "val"],
        help="train/validate/test the model or inference on a video",
    )

    parser.add_argument(
        "-d", "--dataset",
        type=str,
        default="tennis",
        choices=["tennis", "fs_comp", "fs_perf", "FineDiving", "FineGym"],
        help="dataset to train/validate/test on",
    )

    parser.add_argument(
        "--opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="optional configurations to overwrite the configurations in yaml"
    )

    return parser.parse_args()


def parse_yaml_config(yaml_path: Path) -> ConfigNode:
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

    yaml_config = ConfigNode(init_dict=raw_content)
    return yaml_config


def merge_cmd_config(args: argparse.Namespace, yaml_config: ConfigNode) -> ConfigNode:
    config = copy.deepcopy(yaml_config)

    # set dataset
    mode = args.mode
    config[mode]["dataset"] = args.dataset

    def to_value(expression: str):
        # if is not str expression
        if not isinstance(expression, str):
            return expression

        # eval the expression, i.e., 1e-4(str) to 1e-4(float)
        with contextlib.suppress(ValueError, SyntaxError):
            v = literal_eval(expression)
        return v

    # merge opts
    if args.opts is not None:
        opts: list = args.opts
        assert len(opts) % 2 == 0, "option and value mismatch for --opts"

        for full_key, value in zip(opts[::2], opts[1::2]):
            namespace, key = full_key.split(".")
            value = to_value(value)

            # TODO: add tuple-list and list-tuple conversion support
            if isinstance(value, (list, tuple)):
                raise NotImplementedError
            # f"Conversion form {type(value)} is not implemented yet"

            config[namespace][key] = value

    return config


def get_config(yaml_path: Path) -> ConfigNode:
    return merge_cmd_config(get_args(), parse_yaml_config(yaml_path))


if __name__ == "__main__":
    # yaml_config = parse_yaml_config(
    #     Path(__file__).parent.joinpath("../config/vlh.yaml"))

    # yaml_config.to_yaml(Path("./run_config.yaml"))

    print(get_config(Path(__file__).parent.joinpath("../config/vlh.yaml")))
