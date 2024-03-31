# Standard Library
import json
from pathlib import Path
from typing import TypedDict, NamedTuple

# Third-Party Library
import cv2
import numpy as np
import pandas as pd

# My Library
from .csv_utils import parse_csv
from .color import green, red


class Event(TypedDict):
    # The frame when the event starts
    frame: int
    # The label of the event
    label: str
    # The comment of the event
    comment: str


class SourceInfo(TypedDict):
    end_frame: int
    start_frame: int
    pad: list[int]
    effective_pad: list[int]


class Annotation(TypedDict):
    # FPS of the clip
    fps: int
    # Height of the clip
    height: int
    # Width of the clip
    width: int
    # Name of the clip
    video: str
    # Num event in the clip, also the frame dir name
    num_events: int
    # Num frames of the clip
    num_frames: int
    # Annotations of the events
    events: list[Event]
    # Source Information
    _source_info: SourceInfo


class Task(NamedTuple):
    video_name: str
    video_path: Path
    frame_out_path: Path
    min_frame: int
    max_frame: int
    target_fps: float
    target_num_frames: int
    width: int
    height: int
    max_height: int


def get_annotations(annotation_dir: Path) -> list[Annotation]:

    all_annotations = []
    for split in ["train", "test", "val"]:
        split_annotation_file = annotation_dir / f"{split}.json"

        with split_annotation_file.open(mode="r") as f:
            all_annotations.extend(json.load(f))

    return all_annotations


def get_yt_id_by_name(csv: pd.DataFrame, name: str) -> str:
    try:
        yt_id = csv.loc[name, "yt_id"]
    except Exception:
        print(f"{red(f'yt_id not found: {yt_id}')}")
    return yt_id


def get_video_path(indir: Path, prefix: str) -> Path:
    for file in indir.glob("*.mp4"):
        if file.name.startswith(prefix):
            return file
    raise ValueError(
        f"Video {red('not found')} for given youtube ID: {
            green(prefix, True)} at {green(indir)}"
    )


def get_FineGym_tasks(indir: Path, outdir: Path, max_height: int) -> list[Task]:

    annotation_dir = Path(__file__).parent.parent / "FineGym"

    csv = parse_csv(
        annotation_dir / "valid-videos.csv", fields=("name", "yt_id")
    ).set_index("name")

    annotations: list[Annotation] = get_annotations(annotation_dir)

    tasks: list[Task] = []
    for ann in annotations:
        # get video name
        base_video_name = ann["video"]

        # get frame output path
        frame_out_path = outdir / base_video_name

        # get yt_id
        video_name = base_video_name.split("_E_")[0]
        yt_id = get_yt_id_by_name(csv, video_name)

        # get video file path
        video_path = get_video_path(indir, yt_id)

        # get num_frames
        num_frames = ann["num_frames"]

        # get start frame and end frame
        src_info = ann["_source_info"]

        # get fps
        ann_fps = ann["fps"]

        # some of the videos directly downloaded from the Youtube is 30 fps, which may mismatches the original videos from the annotations
        # so the start_frame and end_frame in src_info is wrong
        # we need to recalculated the start_frame and end_frame according to the downloaded fps

        # get the downloaded video fps
        video_capture = cv2.VideoCapture(str(video_path))
        downloaded_fps = video_capture.get(cv2.CAP_PROP_FPS)

        # get the start time and and time according to the video_name
        # video_name: rrrgsW--AE8_E_000510_000574
        start_time, end_time = [
            int(i) for i in base_video_name.split("_E_")[1].split("_")
        ]
        # plus 1 to save the frame of the last second
        # end_time += 1

        # calculate the aligned start frame
        start_frame = downloaded_fps * start_time - src_info["pad"][0]
        end_frame = downloaded_fps * end_time + src_info["pad"][1]

        t = Task(
            video_name=base_video_name,
            video_path=video_path,
            frame_out_path=frame_out_path,
            min_frame=int(start_frame),
            max_frame=int(end_frame),
            target_fps=ann_fps,
            target_num_frames=num_frames,
            width=ann["width"],
            height=ann["height"],
            max_height=max_height,
        )

        tasks.append(t)
    return tasks


def get_tennis_tasks(indir: Path, outdir: Path, max_height: int):

    annotation_dir = Path(__file__).parent.parent / "tennis"

    csv = parse_csv(
        annotation_dir / "valid-videos.csv", fields=("name", "yt_id")
    ).set_index("name")

    annotations: list[Annotation] = get_annotations(annotation_dir)

    tasks: list[Task] = []
    for ann in annotations:
        # get video name
        video_name = ann["video"]

        # get frame output path
        frame_out_path = outdir / video_name

        # get yt_id, start_frame, end_frame
        video_name, start_frame, end_frame = video_name.rsplit("_", 2)
        yt_id = get_yt_id_by_name(csv, video_name)

        # get video file path
        video_path = get_video_path(indir, yt_id)

        # get frames
        num_frames = ann["num_frames"]
        start_frame, end_frame = int(start_frame), int(end_frame)

        if (got_frames := end_frame - start_frame) != num_frames:
            raise ValueError(
                f"Frames mismatch, expected {num_frames}, got {got_frames}"
            )

        # get fps
        fps = ann["fps"]

        t = Task(
            video_name=video_name,
            video_path=video_path,
            frame_out_path=frame_out_path,
            min_frame=start_frame,
            max_frame=end_frame,
            target_fps=fps,
            target_num_frames=num_frames,
            width=ann["width"],
            height=ann["height"],
            max_height=max_height,
        )

        tasks.append(t)
    return tasks


def get_fs_comp_tasks(indir: Path, outdir: Path, max_height: int) -> list[Task]:

    annotation_dir = Path(__file__).parent.parent / "fs_comp"

    csv = parse_csv(
        annotation_dir / "valid-videos.csv", fields=("name", "yt_id")
    ).set_index("name")

    annotations: list[Annotation] = get_annotations(annotation_dir)

    tasks: list[Task] = []
    for ann in annotations:
        # get video name
        video_name = ann["video"]

        # get frame output path
        frame_out_path = outdir / video_name

        # get yt_id, start_frame, end_frame
        video_name, _, start_frame, end_frame = video_name.rsplit("_", 3)
        yt_id = get_yt_id_by_name(csv, video_name)

        # get video file path
        video_path = get_video_path(indir, yt_id)

        # get frames
        num_frames = ann["num_frames"]
        start_frame, end_frame = int(start_frame), int(end_frame)

        if (got_frames := end_frame - start_frame) != num_frames:
            raise ValueError(
                f"Frames mismatch, expected {num_frames}, got {got_frames}"
            )

        # get fps
        fps = ann["fps"]

        t = Task(
            video_name=video_name,
            video_path=video_path,
            frame_out_path=frame_out_path,
            min_frame=start_frame,
            max_frame=end_frame,
            target_fps=fps,
            target_num_frames=num_frames,
            width=ann["width"],
            height=ann["height"],
            max_height=max_height,
        )

        tasks.append(t)
    return tasks


def get_tasks(dataset: str, indir: Path, outdir: Path, max_height: int) -> list[Task]:
    """
    Returns a list of Task for further extracting based on the specified dataset.

    Args:
        dataset: The name of the dataset to extract.
        indir: The directory path containing videos.
        outdir: The directory path to save the extracted frames.
        max_height: The maximum height value.

    Returns:
        A list of Task for the next extracting.

    Raises:
        NotImplementedError: If the dataset is not implemented.
    """
    if dataset == "tennis":
        return get_tennis_tasks(indir, outdir, max_height)
    elif dataset == "FineGym":
        return get_FineGym_tasks(indir, outdir, max_height)
    elif dataset == "fs_comp":
        return get_fs_comp_tasks(indir, outdir, max_height)
    elif dataset == "FineDiving":
        return []
    else:
        raise NotImplementedError


if __name__ == "__main__":
    get_FineGym_tasks(
        Path("/data/wsf/projects/VideoLabelHierarchy/datasets/FineGym99/FineGym99"),
        Path("/data/wsf/projects/video-datasets/FineGym-frames"),
        224,
    )

    get_tennis_tasks(
        Path("/data/wsf/projects/video-datasets/tennis"),
        Path("/data/wsf/projects/video-datasets/tennis-frames"),
        224,
    )
