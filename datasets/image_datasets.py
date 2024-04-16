# Standard Library
import copy
import random
from pathlib import Path
from typing import Optional

# Third-Party Library
import numpy as np

# Torch Library
import torch
import torch.utils.data as data

# My Library
from .frame import get_frame_reader

from utils.color import error, red
from utils.io import load_json, load_yaml
from utils.annotation import Annotation, HierarchalClass, Example


def get_classes(class_txt: Path) -> dict[str, int]:
    with class_txt.open(mode="r") as f:
        classes: list[str] = f.readlines()
    # zero for background
    return {c.strip(): i + 1 for i, c in enumerate(classes)}


# TODO: E2E-Spot给的FineDiving的label有问题, 只给了细分的动作, 没有207c这样的动作, 所以后面得考虑一下怎么做
# TODO: 目前是根据给定的label分成了入水, 转体, 屈体, 抱膝, 参考: https://www.diving-concepts.com/blank-5
def get_hierarchal_classes(class_yaml: Path) -> HierarchalClass:
    content = load_yaml(class_yaml)

    num_level = content["meta"]["level"]
    num_trans = content["meta"]["transition"]

    level_dict = {}
    for level_idx in range(num_level):
        level_class = content["classes"][f"level{level_idx + 1}"]
        level_dict[level_idx] = {
            cls_name: i + 1 for i, cls_name in enumerate(level_class)}

    trans_dict = {
        trans_idx: content["transition"][f"transition{trans_idx + 1}"]
        for trans_idx in range(num_trans)
    }
    return {"num_level": num_level, "num_trans": num_trans, "level_dict": level_dict, "trans_dict": trans_dict}


class ActionSpotDataset(data.Dataset):
    """ dataset for training, validating and testing """

    def __init__(
        self,
        # Dict of hierarchal classes
        hierarchal_classes: HierarchalClass,
        # Path to the label json
        label_file: Path,
        # Path to the frames
        clip_dir: Path,
        # Modality of the frames, [rgb, bw, flow]
        modality: str,
        # Length of the clip, i.e. frame num
        clip_len: int,
        # Length of the datasets, i.e., clip num
        dataset_len: int,
        # Disable random augmentations to each frame in a clip
        is_eval: bool = True,
        # Dimension to crop the clip
        crop_dim: Optional[int] = None,
        # Stride to sample the clip, >1 to downsample the video
        frame_sample_stride: int = 1,
        # If apply the same random crop augmentation to each frame in a clip
        same_crop_transform: bool = True,
        # Dilate ground truth labels
        dilate_len: int = 0,
        mixup: bool = False,
        # Number of the frames to pad before/after the clip
        n_pad_frames: int = 5,
        # Sample event ratio
        event_sample_rate: float = -1
    ) -> None:
        super().__init__()

        # save the parameters
        self.label_file: Path = label_file
        self.clip_labels: list[Annotation] = load_json(
            label_file)
        self.hierarchal_classes: HierarchalClass = hierarchal_classes
        self.clip_indexes: dict[str, int] = {
            x["video"]: i for i, x in enumerate(self.clip_labels)}
        self.is_eval = is_eval
        self.dilate_len = dilate_len
        self.event_sample_rate = event_sample_rate
        self.mixup = mixup

        # parameters that need verify
        self.clip_len: int = clip_len
        assert clip_len > 0, error(
            f"clip len should be greater than 0, got: {red(clip_len, True)}")

        self.frame_sample_stride: int = frame_sample_stride
        assert frame_sample_stride > 0, error(
            f"frame_sample_stride should be greater than 0, got: {red(frame_sample_stride, True)}")

        self.dataset_len = dataset_len
        assert dataset_len > 0, error(
            f"dataset_len should be greater than 0, got: {red(dataset_len, True)}")

        self.n_pad_frames = n_pad_frames
        assert n_pad_frames >= 0, error(
            f"n_pad_frames should be greater equal than 0, got: {red(n_pad_frames, True)}")

        # Sample based on foreground labels
        if self.event_sample_rate > 0:
            self.flat_labels = []
            for i, x, in enumerate(self.clip_labels):
                for event in x["events"]:
                    if event["frame"] < x["num_frames"]:
                        self.flat_labels.append((i, event["frame"]))

        # Sample based on the clip length
        num_frames = [c["num_frames"] for c in self.clip_labels]
        self.uniform_sample_weight = np.array(num_frames) / np.sum(num_frames)

        # Frame Reader
        self.frame_reader = get_frame_reader(
            clip_dir=clip_dir, is_eval=is_eval, modality=modality, crop_dim=crop_dim,
            same_crop_transform=same_crop_transform, multi_crop=False)

    def sample_clip(self) -> tuple[Annotation, int]:
        """
        Uniformly samples a clip label and start frame based on specified parameters.

        Returns:
            tuple[Annotation, int]: A tuple containing the sampled clip label and start frame.
        """
        clip_label = random.choices(
            self.clip_labels, weights=self.uniform_sample_weight)[0]

        clip_frames = clip_label["num_frames"]
        # every time we sample a same clip, we would like it having some frame-shifting, i.e.
        # the first time we sample clip A from frame 0 to frame 100
        # the next time we sample clip A again, we would like it from 10-110
        # so with some frame-shifting, we increase the total amount of training examples
        start_frame = -self.n_pad_frames * self.frame_sample_stride + random.randint(
            0, max(0, clip_frames - 1 + (2 * self.n_pad_frames - self.clip_len) * self.frame_sample_stride))
        return clip_label, start_frame

    def sample_event(self) -> tuple[Annotation, int]:
        """
        Uniformly samples a event label and start frame based on specified parameters.

        Returns:
            tuple[Annotation, int]: A tuple containing the sampled event label and start frame.
        """
        video_idx, frame_idx = random.choices(self.flat_labels)[0]
        clip_label = self.clip_labels[video_idx]
        video_len = clip_label['num_frames']

        lower_bound = max(
            -self.n_pad_frames * self.frame_sample_stride,
            frame_idx - self.clip_len * self.frame_sample_stride + 1)
        upper_bound = min(
            video_len - 1 + (self.n_pad_frames - self.clip_len) *
            self.frame_sample_stride,
            frame_idx)

        start_frame = random.randint(lower_bound, upper_bound) \
            if upper_bound > lower_bound else lower_bound

        assert start_frame <= frame_idx
        assert start_frame + self.clip_len > frame_idx
        return clip_label, start_frame

    def get_sample(self) -> tuple[Annotation, int]:
        # because event is rarely sparse in the clip, so we need to sample
        # the event to increase the training examples
        if self.event_sample_rate > 0 and random.random() > self.event_sample_rate:
            clip_label, start_frame = self.sample_event()
        else:
            clip_label, start_frame = self.sample_clip()
        return clip_label, start_frame

    def get_example(self) -> Example:
        clip_label, start_frame = self.get_sample()

        # build hierarchal labels
        labels = {level_idx: np.zeros(self.clip_len) for level_idx in range(
            self.hierarchal_classes["num_level"])}

        for event in clip_label["events"]:
            event_frame = event["frame"]

            # calculate the index of the frame
            label_idx = (
                event_frame - start_frame) // self.frame_sample_stride
            if (label_idx >= -self.dilate_len and label_idx < self.clip_len + self.dilate_len):

                # modify the corresponding frame labels
                for i in range(max(0, label_idx - self.dilate_len), min(self.clip_len, label_idx + self.dilate_len + 1)):

                    # label of the first level
                    level0_label = event["label"]

                    # build the label for the first level
                    labels[0][i] = self.hierarchal_classes["level_dict"][0][level0_label]

                    # build the label for the rest level
                    last_level_label: str = level0_label
                    for level_idx in range(1, self.hierarchal_classes["num_level"]):
                        level_label: str = ""
                        # get the label of current level
                        for j in range(level_idx):
                            level_label = self.hierarchal_classes["trans_dict"][j][last_level_label]
                        labels[level_idx][i] = self.hierarchal_classes["level_dict"][level_idx][level_label]

        # load frames
        frames = self.frame_reader.load_frames(
            clip_name=clip_label["video"], start_frame=start_frame,
            end_frame=start_frame + self.clip_len * self.frame_sample_stride,
            pad_end_frame=True, frame_sample_stride=self.frame_sample_stride, random_sample=not self.is_eval
        )

        return {"frame": frames, "level": self.hierarchal_classes["num_level"], "label": labels, "contains_event": int(labels[0].sum() > 0)}

    def __getitem__(self, unused) -> Example:
        """
        Return: 
            {
                "frame": torch.FloatTensor, shape [Temporal, Channel, Height, Width],
                "level": int, level of the labels of the example,
                "label": list[torch.FloatTensor], labels of each level, each shape [Temporal]
                "contains_event": bool, if contains event
            }
        """
        example = self.get_example()
        while example["frame"] is None:
            example = self.get_example()
        return example

    def __len__(self):
        return self.dataset_len


class ActionSpotmAPDataset(data.Dataset):
    """ dataset for calculating mAP """

    def __init__(
        self,
        # Path to the label json
        label_file: Path,
        # Path to the frames
        clip_dir: Path,
        # Modality of the frames, [rgb, bw, flow]
        modality: str,
        # Length of the clip, i.e. frame num
        clip_len: int,
        # overlap of each clip fragment
        overlap_len: int = 0,
        # Dimension to crop the clip
        crop_dim: Optional[int] = None,
        # Stride to sample the clip, >1 to downsample the video
        frame_sample_stride: int = 1,
        # Number of the frames to pad before/after the clip
        n_pad_frames: int = 5,
        # If multi-crop on images
        multi_crop: bool = False,
        # If horizontally flip the clip
        horizontal_flip: bool = False,
        # TODO: add doc
        skip_partial_end: bool = True
    ) -> None:
        super().__init__()

        # save the parameters
        self.label_file: Path = label_file
        self.clip_labels: list[Annotation] = load_json(label_file)
        self.clip_indexes: dict[str, int] = {
            x["video"]: i for i, x in enumerate(self.clip_labels)}
        self.clip_len: int = clip_len

        # parameters that need verification
        assert frame_sample_stride > 0
        self.frame_sample_stride: int = frame_sample_stride

        # augmentations used in testing
        self.horizontal_flip = horizontal_flip
        self.multi_crop = multi_crop

        # Frame Reader
        self.frame_reader = get_frame_reader(
            clip_dir=clip_dir, is_eval=True, crop_dim=crop_dim, modality=modality, same_crop_transform=True, multi_crop=multi_crop
        )

        # split clips into smaller clips for testing
        self.clips: list[dict[str, str | int]] = []
        for ann in self.clip_labels:
            has_clip = False
            for i in range(
                -n_pad_frames * frame_sample_stride,
                max(0, ann["num_frames"] - overlap_len *
                    frame_sample_stride * int(skip_partial_end)),
                (clip_len - overlap_len) * frame_sample_stride
            ):
                has_clip = True
                self.clips.append(
                    {"clip_name": ann["video"], "start_frame": i})
            assert has_clip, i

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, index: int) -> dict[str, str | int | torch.FloatTensor]:
        """ returned frame is [B, T, C, H, W] if use multi-crop or horizontal flip, else [T, C, H, W]  """
        clip_name, start_frame = self.clips[index].values()

        # load_frames
        frames = self.frame_reader.load_frames(
            clip_name, start_frame, start_frame + self.clip_len * self.frame_sample_stride,
            pad_end_frame=True,
            frame_sample_stride=self.frame_sample_stride
        )

        if self.horizontal_flip:
            frames = torch.stack([frames, frames.flip(-1)], dim=0)

        return {"clip_name": clip_name, "start_frame": start_frame // self.frame_sample_stride, "frame": frames}

    @property
    def augmentation(self) -> bool:
        """ if dataset uses augmentation """
        return self.horizontal_flip or self.multi_crop

    @property
    def videos(self) -> list[tuple[str, int, float]]:
        return sorted([(v["video"], v["num_frames"] // self.frame_sample_stride, v["fps"] / self.frame_sample_stride) for v in self.clip_labels])

    @property
    def labels(self) -> list[Annotation]:
        if self.frame_sample_stride == 1:
            return self.clip_labels

        labels = []
        for ann in self.clip_labels:
            ann_copy = copy.deepcopy(ann)
            ann_copy["fps"] /= self.frame_sample_stride
            ann_copy["num_frames"] //= self.frame_sample_stride
            for event in ann_copy["events"]:
                event["frame"] //= self.frame_sample_stride
            labels.append(ann_copy)
        return labels


if __name__ == "__main__":
    # import pprint

    # pprint.pprint(get_hierarchal_classes(
    #     Path(__file__).resolve().parent / "../tools/tennis/class.yaml"))

    import torch.utils.data as data

    from utils.config import parse_yaml_config

    config = parse_yaml_config(
        Path(__file__).parent.joinpath("../config/vlh.yaml"))

    base_dir = Path(config["variables"]["basedir"])

    for dataset_name, dataset_config in config["datasets"].items():

        clip_dir = Path(dataset_config["clip_dir"])
        class_file = Path(dataset_config["class_file"])

        hierarchal_classes = get_hierarchal_classes(class_file)

        for split, clip_len, dataset_len, is_eval, same_crop_transform in [
            ("train", 100, 50000, False, False),
            ("val", 150, 50000 // 4, True, True),
            ("test", 200, 50000 // 4, True, True),
        ]:
            dataset = ActionSpotDataset(
                hierarchal_classes=hierarchal_classes,
                label_file=base_dir.joinpath(
                    f"tools/{dataset_name}/{split}.json"),
                clip_dir=clip_dir,
                modality="rgb",
                clip_len=clip_len,
                dataset_len=dataset_len,
                is_eval=is_eval,
                same_crop_transform=same_crop_transform,
                crop_dim=100 if split == "test" else 200
            )

            loader = data.DataLoader(dataset, batch_size=1)

            print(f"test {dataset_name}, {split}")

            for example in loader:
                if not (example["label"][0] == 0).all():
                    print(example["contains_event"])
                    print(example["level"])
                    print(example["frame"].shape,
                          example["frame"].size(1) == 100)
                    for j in range(example["level"]):
                        print(example["label"][j].shape,
                              example["label"][j].size(1) == 100)
                    break

            if split == "train":
                continue

            map_dataset = ActionSpotmAPDataset(
                label_file=base_dir.joinpath(
                    f"tools/{dataset_name}/{split}.json"),
                clip_len=clip_len,
                clip_dir=clip_dir,
                modality="rgb",
                overlap_len=10,
            )

            map_loader = data.DataLoader(map_dataset, batch_size=1)

            print(f"test {dataset_name}, {split}, mAPDataset")
            for example in map_loader:
                print(example["clip_name"])
                print(example["start_frame"])
                print(example["frame"].shape)
                break
