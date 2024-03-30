# Standard Library
from typing import Callable

# Third-Party Library
from colorama import Fore, Style, init

def colorizer(color: int) -> Callable:
    def decorator(func: Callable) -> Callable:
        def wrapper(text: str, bright: bool = False) -> str:
            colored_text = f"{color}{Style.BRIGHT if bright else ''}{
                text}{Style.RESET_ALL}"
            return colored_text
        return wrapper
    return decorator


@colorizer(Fore.GREEN)
def green():
    ...


@colorizer(Fore.YELLOW)
def yellow():
    ...


@colorizer(Fore.RED)
def red():
    ...
