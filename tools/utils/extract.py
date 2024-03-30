# Standard Library
from pathlib import Path

# Third-Party Library
import av
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

# My Library
from .tasks import Task

cv2.setNumThreads(1)


def extract_frames_tennis(task: Task):
    # capture video
    vc = cv2.VideoCapture(str(task.video_path))

    # get video properties
    fps = vc.get(cv2.CAP_PROP_FPS)
    exp_num_frames = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Assert that the version of the videos is the same
    assert (w, h) == (
        task.width,
        task.height,
    ), f'Mismatch in original frame dimensions: {task.video_name} -- got {(w, h)}, expected {(task.width, task.height)}, {task.video_name}'

    # resize the frame if necessary
    if task.max_height < h:
        oh = task.max_height
        ow = int(w / h * task.max_height)
    else:
        oh, ow = h, w

    # skip if the clip is already extracted
    file_count = sum(1 for _ in task.frame_out_path.glob("*.jpg"))
    if file_count == task.target_num_frames:
        # this breaks tqdm bar
        # print(f"Already extracted, skip: {task.video_name}")
        return

    # check the frames
    assert np.isclose(fps, task.target_fps, atol=0.01), f"target FPS {
        task.target_fps} does not match source FPS {fps}, {task.video_name}"

    # if dry run
    if task.frame_out_path is not None:
        task.frame_out_path.mkdir(exist_ok=True, parents=True)

    vc.set(cv2.CAP_PROP_POS_FRAMES, task.min_frame)
    i = 0
    while True:
        ret, frame = vc.read()
        if not ret:
            break

        if frame.shape[0] != oh:
            frame = cv2.resize(frame, (ow, oh))

        if task.frame_out_path is not None:
            frame_path = task.frame_out_path / f"{i:06d}.jpg"
            cv2.imwrite(str(frame_path), frame)

        i += 1
        if task.min_frame + i == task.max_frame:
            break

    vc.release()
    assert i == (e := task.max_frame -
                 task.min_frame), f"frames mismatch, expected {e}, got {i}, {task.video_name}"


def extract_frames_fs_comp(task: Task):
    # capture video
    vc = cv2.VideoCapture(str(task.video_path))

    # get video properties
    fps = vc.get(cv2.CAP_PROP_FPS)
    exp_num_frames = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # resize the frame if necessary
    if task.max_height < h:
        oh = task.max_height
        ow = int(w / h * task.max_height)
    else:
        oh, ow = h, w

    # skip if the clip is already extracted
    file_count = sum(1 for _ in task.frame_out_path.glob("*.jpg"))
    if file_count == task.target_num_frames:
        # this breaks tqdm bar
        # print(f"Already extracted, skip: {task.video_name}")
        return

    # check the frames
    assert np.isclose(fps, task.target_fps, atol=0.01), f"target FPS {
        task.target_fps} does not match source FPS {fps}, {task.video_name}"

    # if dry run
    if task.frame_out_path is not None:
        task.frame_out_path.mkdir(exist_ok=True, parents=True)

    vc.set(cv2.CAP_PROP_POS_FRAMES, task.min_frame)
    i = 0
    while True:
        ret, frame = vc.read()
        if not ret:
            break

        if frame.shape[0] != oh:
            frame = cv2.resize(frame, (ow, oh))

        if task.frame_out_path is not None:
            frame_path = task.frame_out_path / f"{i:06d}.jpg"
            cv2.imwrite(str(frame_path), frame)

        i += 1
        if task.min_frame + i == task.max_frame:
            break

    vc.release()
    assert i == (e := task.max_frame -
                 task.min_frame), f"frames mismatch, expected {e}, got {i}, {task.video_name}"


def extract_frames_finegym(task: Task):
    # capture video
    vc = cv2.VideoCapture(str(task.video_path))

    # get video properties
    fps = vc.get(cv2.CAP_PROP_FPS)
    exp_num_frames = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Assert that the version of the videos is the same
    assert (w, h) == (
        task.width,
        task.height,
    ), f'Mismatch in original frame dimensions: {task.video_name} -- got {(w, h)}, expected {(task.width, task.height)}, {task.video_name}'

    # resize the frame if necessary
    if task.max_height < h:
        oh = task.max_height
        ow = int(w / h * task.max_height)
    else:
        oh, ow = h, w

    # skip if the clip is already extracted
    file_count = sum(1 for _ in task.frame_out_path.glob("*.jpg"))
    if file_count == task.target_num_frames:
        # this breaks tqdm bar
        # print(f"Already extracted, skip: {task.video_name}")
        return

    # if dry run
    if task.frame_out_path is not None:
        task.frame_out_path.mkdir(exist_ok=True, parents=True)

    # check the frames
    if np.isclose(fps, task.target_fps, atol=0.01):
        # downloaded fps = 30, target_fps = 30
        downsample = False
    else:
        # downloaded fps = 30, target_fps = 29.49
        downsample = True
        num_frame = task.max_frame - task.min_frame
        skip_interval = num_frame // (num_frame - task.target_num_frames)

    vc.set(cv2.CAP_PROP_POS_FRAMES, task.min_frame)
    i = 0
    while True:
        ret, frame = vc.read()
        if not ret:
            break

        if downsample and (i % skip_interval == 0):
            i += 1
            continue

        if frame.shape[0] != oh:
            frame = cv2.resize(frame, (ow, oh))

        if task.frame_out_path is not None:
            frame_path = task.frame_out_path / f"{i:06d}.jpg"
            cv2.imwrite(str(frame_path), frame)

        i += 1
        if task.min_frame + i == task.max_frame:
            break

    vc.release()
    assert i == (e := task.max_frame -
                 task.min_frame), f"frames mismatch, expected {e}, got {i}, {task.video_name}"


if __name__ == "__main__":
    from .tasks import get_FineGym_tasks

    tasks = get_FineGym_tasks(
        Path("/data/wsf/projects/VideoLabelHierarchy/datasets/FineGym99/FineGym99"),
        Path("/data/wsf/projects/video-datasets/FineGym-frames"),
        224,
    )

    task = None
    for t in tasks:
        if t.video_name == "yj0pNXcTK0k":
            task = t

    # tasks = get_tasks("tennis", Path("/data/wsf/projects/video-datasets/tennis"), Path(__file__).parent.parent.parent / "second", 224)[0]

    task = Task(
        video_name="yj0pNXcTK0k",
        video_path=Path(
            "/data/wsf/projects/video-datasets/FineGym/yj0pNXcTK0k.2018 Cottbus World Cup Day 1 Olympic Channel.mp4"),
        frame_out_path=Path(__file__).resolve().parent / "../../second",
        min_frame=10067,
        max_frame=10872,
        target_fps=29.97,
        target_num_frames=760,
        width=1280,
        height=720,
        max_height=224,)

    extract_frames_finegym(task=task)
