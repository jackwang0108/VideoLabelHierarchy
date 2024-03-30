# Standard Library
import os
from pathlib import Path

# Third-Party Library
from yt_dlp import YoutubeDL

# My Library
from .color import green, red, yellow


def download(
    outdir: str | Path,
    ip: str,
    port: int,
    yt_id: str,
    fps: int,
    height: str,
    width: str,
) -> tuple[bool, str]:
    """
    使用指定参数调用yt-dlp下载YouTube视频。

    参数:
        args (tuple): 包含以下元素的元组，按顺序排列：
            - outdir (str): 下载视频的输出目录。
            - ip (str): 代理连接的IP地址。
            - port (int): 代理连接的端口号。
            - youtube_id (str): YouTube视频的ID。
            - fps (int): 所需的每秒帧数。
            - height (str): 视频的所需高度。
            - width (str): 视频的所需宽度。

    返回:
        tuple: 包含下载结果和下载的视频youtube_id的元组。
    """

    # get download options
    opts = {
        "quiet": True,
        "noprogress": True,
        "fixup": "detect_or_warn",
        "geo_bypass_country": "US",
        "outtmpl": f"{outdir}/%(id)s.%(title)s.%(ext)s",
        "ffmpeg_location": os.environ["ffmpeg_path"],
        "format": f"bv*[width={width}][height={height}][fps={fps}][ext=mp4][protocol=https]",
    }

    use_proxy = ip is not None and port is not None
    if use_proxy:
        opts["proxy"] = f"http://{ip}:{port}"

    # print downloading args
    info = [
        "yt-dlp",
        f"-P {green('/'.join(outdir.parts[-2:]), True)}",
        ("--proxy" if use_proxy else ""),
        (f"http://{green(ip, True)}:{green(port, True)}" if use_proxy else ""),
        "-f",
        f"bv*[width={green(width, True)}]"
        + f"[height={green(height, True)}]"
        + f"[fps={green(fps, True)}]"
        + f"[ext={green('mp4', True)}]",
        f"https://www.youtube.com/watch?v={green(yt_id, True)}",
    ]
    print(" ".join(info))

    yt = YoutubeDL(opts)
    url = f"https://www.youtube.com/watch?v={yt_id}"

    try:
        result = yt.download(url)
    except Exception:
        result = 1
        print(f"Download video {red(f'{yt_id} failed')}.")
    return result, yt_id
