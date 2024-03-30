# Standard Library
import requests


def get_proxy_handler(port: int | str = 7890, ip: str = "127.0.0.1") -> dict[str, str]:
    proxy_handler: dict[str, str] = {
        "http": f"http://{ip}:{port}",
        "https": f"http://{ip}:{port}",
    }
    return proxy_handler


def valid_proxy(port: int | str, ip: str = "127.0.0.1") -> bool:
    proxy_handler = get_proxy_handler(port=port, ip=ip)

    succeeded = False
    try:
        response = requests.get("https://www.google.com", proxies=proxy_handler)
        succeeded = response.status_code == 200
    except requests.exceptions.ConnectionError:
        succeeded = False
    return succeeded
