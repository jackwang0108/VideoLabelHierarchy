# Standard Library
import os
import shutil
import argparse
import multiprocessing
from pathlib import Path

# My Library
from .utils.color import red, green
from .utils.download import download
from .utils.csv_utils import parse_csv
from .utils.proxy import valid_proxy, get_proxy_handler


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Command-Line tool for downloading video datasets"
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        required=True,
        choices=["tennis", "FineDiving", "FineGym", "fs_comp", "fs_perf"],
        help="Which dataset to download",
    )
    parser.add_argument(
        "-f",
        "--ffmpeg",
        default="ffmpeg",
        type=str,
        help="Path of ffmpeg executable file",
    )
    parser.add_argument(
        "-o",
        "--outdir",
        default="None",
        type=str,
        help="path to save downloaded videos",
    )
    parser.add_argument(
        "-p", "--port", default=7890, type=int, help="port of command-line proxy"
    )
    parser.add_argument(
        "-i", "--ip", default="127.0.0.1", type=str, help="ip of command-line proxy"
    )
    parser.add_argument(
        "-n",
        "--num_proc",
        type=int,
        default=multiprocessing.cpu_count() // 2,
        help="num process to use",
    )
    return parser.parse_args()


def main(args: argparse.Namespace):
    assert args.dataset in (
        ds := ["tennis", "FineDiving", "FineGym", "fs_comp", "fs_perf"]
    ), f"args.dataset should be in {green(ds)}, got: {red(args.dataset)}"

    ip: str = args.ip
    ffmpeg: str = args.ffmpeg
    port: int = int(args.port)
    dataset: str = args.dataset

    # num process
    n_proc = int(args.num_proc)
    assert n_proc < (c := multiprocessing.cpu_count()), red(
        f"too many process, you only got {c} cpus"
    )

    # save dir
    outdir: Path = Path(args.outdir) if args.outdir != "None" else Path(f"./{dataset}")
    if not outdir.exists():
        outdir.mkdir(parents=True)
    print(f"save vidoes to {green(outdir)}")

    # ffmpeg
    assert shutil.which(ffmpeg) is not None or Path(ffmpeg).exists(), red(
        "ffmpeg not found!", True
    )
    os.environ["ffmpeg_path"] = ffmpeg
    print(f"set ffmpeg path to: {green(ffmpeg)}")

    # proxy
    if ip is not None and port is not None:
        print(f"proxy provided: {green(ip)}:{green(port)}")
        proxy_status = valid_proxy(port=port, ip=ip)
        if not proxy_status:
            print(red(f"Proxy test {red('failed', True)} for https://{ip}:{port}"))
            return False
        print(f"Proxy test {green('success')} for https://{ip}:{port}")
    else:
        print("proxy not provided, skipping...")

    # get videos
    tasks = parse_csv(
        Path(__file__).parent / dataset / "valid-videos.csv",
        ("yt_id", "fps", "height", "width"),
    ).values.tolist()

    # multiprocessing download
    async_result = []
    pool = multiprocessing.Pool(n_proc)
    for t in tasks:
        yt_id, fps, height, width = t
        async_result.append(
            pool.apply_async(download, (outdir, ip, port, yt_id, fps, height, width))
        )
    pool.close()
    pool.join()

    # log result
    results = [i.get() for i in async_result]
    with Path(f"./{args.dataset}-result.txt").open(mode="w") as f:
        for r in results:
            success, yt_id = r
            success = success == 0
            print(
                f"youtube ID: {yt_id}, {green('success') if success else red('fail', True)}"
            )
            f.write(f"youtube ID: {yt_id}, {'success' if success else 'fail'}\n")


if __name__ == "__main__":
    main(get_args())
