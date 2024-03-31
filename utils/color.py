# Standard Library
from typing import Union, Callable


# Third-Party Library
from colorama import Fore, Style, init

init(autoreset=True)

Printable = Union[str, int, float, bool, list, tuple, dict, set, None]


def colorizer(color_code: int, prefix="", bright: bool = False):
    def colorizer(func: Callable) -> Callable:
        def wrapper(text: Printable, _bright: bool = False) -> str:
            colored_text = f"{color_code}{prefix}{Fore.RESET}{color_code}{
                Style.BRIGHT if _bright or bright else ''}{text}{Style.RESET_ALL}"
            return colored_text
        return wrapper
    return colorizer


@colorizer(Fore.BLUE, prefix="DEBUG: ")
def debug():
    ...


@colorizer(Fore.GREEN, prefix="INFO: ")
def info():
    ...


@colorizer(Fore.YELLOW, prefix="WARN: ")
def warn(text: Printable, bright: bool = False):
    ...


@colorizer(Fore.RED, prefix="ERROR: ", bright=True)
def error(text: Printable):
    ...


@colorizer(Fore.BLUE)
def blue():
    ...


@colorizer(Fore.GREEN)
def green():
    ...


@colorizer(Fore.YELLOW)
def yellow():
    ...


@colorizer(Fore.RED, bright=True)
def red():
    ...
