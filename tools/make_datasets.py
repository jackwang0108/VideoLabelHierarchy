# Standard Library
import argparse
import multiprocessing
from pathlib import Path
from typing import Callable
from multiprocessing import Pool

# Third-Party Library
from tqdm import tqdm

# My Library
from .utils.color import green
from .utils.tasks import get_tasks
from .utils.extract import extract_frames_tennis, extract_frames_finegym, extract_frames_fs_comp, copy_frames_finediving

# TODO: 增加SoccerNet和SoccerNet-ball的支持


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Command-Line tool for making video datasets, i.e. extracting the clips from the original video, so as to make preparation for the training"
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        required=True,
        choices=["tennis", "FineDiving", "FineGym", "fs_comp", "fs_perf"],
        help="Which dataset to make",
    )
    parser.add_argument(
        "-i",
        "--indir",
        type=str,
        required=True,
        help="Path to the downloaded videos/frames",
    )
    parser.add_argument(
        "-o",
        "--outdir",
        default=None,
        type=str,
        help="Path to save the extracted clips (None for dry run)",
    )
    parser.add_argument(
        "-m",
        "--max_height",
        type=int,
        default=224,
        help="Max height of the extracted frames",
    )
    parser.add_argument(
        "-n",
        "--num_proc",
        type=int,
        default=multiprocessing.cpu_count() // 2,
        help="Num process to use",
    )
    return parser.parse_args()


def main(args: argparse.Namespace):
    # get args
    dataset = args.dataset
    indir = Path(args.indir)
    outdir = Path(args.outdir)
    max_height = int(args.max_height)
    num_proc = int(args.num_proc)

    print(f"datasets: {green(dataset)}")
    print(f"video indir: {green(indir)}")
    print(f"frame outdir: {green(outdir)}")
    print(f"max_height: {green(max_height)}")
    print(f"num_proc: {green(num_proc)}")

    # if dry run
    is_dry_run = False
    if outdir is None:
        print("No output directory given. Doing a dry run!")
        is_dry_run = True
    else:
        outdir.mkdir(exist_ok=True, parents=True)

    # get tasks
    print("Getting tasks...")
    tasks = get_tasks(dataset, indir, outdir, max_height)

    # get extraction function
    extract_func: Callable
    if dataset == "tennis":
        extract_func = extract_frames_tennis
    elif dataset == "FineGym":
        extract_func = extract_frames_finegym
    elif dataset == "fs_comp":
        extract_func = extract_frames_fs_comp
    elif dataset == "FineDiving":
        copy_func = copy_frames_finediving
    else:
        raise NotImplementedError

    if dataset != "FineDiving":
        with Pool(num_proc) as pool:
            for _ in tqdm(
                pool.imap_unordered(extract_func, tasks),
                total=len(tasks),
                desc="Dry run" if is_dry_run else "Extracting",
            ):
                pass
    else:
        copy_func(indir, outdir, max_height)

    print(f"Done, extracted frames are saved to {green(outdir)}")


if __name__ == "__main__":
    main(get_args())
