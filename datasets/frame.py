# Standard Library
import random
from pathlib import Path
from typing import Optional
from contextlib import nullcontext

# Torch Library
import torch
import torchvision
import torch.nn as nn

# My Library
from .transforms import SameRandomStateContext
from .transforms import get_crop_transform, get_img_transform


class FrameReader:
    """ Read all frames of a clip in a given dataset """

    def __init__(
        self,
        # Dir containing all the clips
        clip_dir: Path,
        # Modality of the frame
        modality: str,
        # Transform of the crop
        crop_transform: torch.ScriptModule,
        # Transform of the image
        img_transform: torch.ScriptModule,
        # If using the same transforms for all image
        same_crop_transform: bool = False,
    ) -> None:
        """
        Initialize the dataset with the specified parameters.

        Args:
            frame_dir: Dir containing all frames of the clip.
            modality: Modality of the clip.
            crop_transform: Transform of the crop.
            img_transform: Transform of the image.
            same_transform: If using the same transforms for all images (default is False).
        """
        self.clip_dir: Path = clip_dir
        self.crop_transform = crop_transform
        self.image_transform = img_transform
        self.same_transform = same_crop_transform
        self.is_flow_img: bool = modality == "flow"

    @staticmethod
    def get_image_name(name: int) -> str:
        return f"{name:06d}.jpg"

    def _read_frame(self, frame_path: Path) -> torch.FloatTensor:
        """ Read an image frame from the given path and return it as a torch.FloatTensor.  """
        img: torch.Tensor = torchvision.io.read_image(
            str(frame_path)).float() / 255
        if self.is_flow_img:
            img = img[1:, :, :]
        return img

    def load_frames(
        self,
        # The name of the video
        clip_name: str,
        # Start frame of the clip
        start_frame: int,
        # End frame of the clip
        end_frame: int,
        # If padding the end frame
        pad_end_frame: bool = False,
        # Sample stride of the clip
        frame_sample_stride: int = 1,
        # If doing random sample
        random_sample: bool = False
    ) -> torch.FloatTensor:
        """
        Load frames from a video clip based on the specified parameters. Return [B, T, C, H, W] if img_transform contains multi-crop else return [T, C, H, W]. T for temporal


        Args:
            video_name: The name of the video.
            start_frame: Start frame of the clip.
            end_frame: End frame of the clip.
            pad_end_frame: If padding the end frame (default is False).
            sample_stride: Sample stride of the clip (default is 1).
            random_sample: If doing random sample (default is False).

        Returns:
            torch.FloatTensor: The loaded frames as a torch tensor.
        """

        # number of padding at the start or end of the clip
        n_pad_end = 0
        n_pad_start = 0

        # containing all read frames to stack
        all_frames: list[torch.FloatTensor] = []

        for frame_idx in range(start_frame, end_frame, frame_sample_stride):
            if random_sample and frame_sample_stride > 1:
                frame_idx += random.randint(0, frame_sample_stride - 1)

            if frame_idx < 0:
                n_pad_start += 1
                continue

            frame_path = self.clip_dir.joinpath(
                clip_name, self.get_image_name(frame_idx))

            try:
                # read the image from the disk
                img = self._read_frame(frame_path=frame_path)

                # crop the image if crop_transform is provided
                with SameRandomStateContext() if self.same_transform else nullcontext():
                    img = self.crop_transform(
                        img) if self.crop_transform else img

                # transforms the image, [B, C, H, W] if multi-crop else [C, H, W]
                img = img if self.same_transform else self.image_transform(img)

                all_frames.append(img)

            except RuntimeError:
                # if a frame is missing, then pad at the end of the tensor
                # print(warn(f"Missing frame {yellow(self.get_image_name(frame_idx))} for video {
                #       yellow(video_name)}, at clip {yellow(frame_path.parent.stem)}, skip this frame..."))
                n_pad_end += 1

        if all_frames:
            # stack the images, [B, T, C, H, W] if multi-crop else [T, C, H, W]
            ret = torch.stack(all_frames, dim=int(all_frames[0].ndim == 4))

            # always padding before the start_frame, and pad end_frame if required
            if n_pad_start > 0 or (pad_end_frame and n_pad_end > 0):
                ret = nn.functional.pad(
                    ret, (0, 0, 0, 0, 0, 0, n_pad_start,
                          n_pad_end if pad_end_frame else 0)
                )
        else:
            ret = None

        return ret


def get_frame_reader(
    clip_dir: Path,
    is_eval: bool,
    modality: str,
    same_crop_transform: bool,
    multi_crop: bool,
    crop_dim: Optional[int] = None,
) -> FrameReader:
    """
    Create and return a FrameReader for reading frames of a clip in a dataset.

    Args:
        clip_dir: Directory containing the clip.
        is_eval: Flag indicating if in evaluation mode.
        crop_dim: Dimension of the crop.
        modality: Modality of the frame.
        same_crop_transform: Flag indicating if using the same crop transform for all images.
        multi_crop: Flag indicating if multi-crop is used.

    Returns:
        FrameReader: A FrameReader for reading frames of a clip.
    """
    crop_transform = get_crop_transform(
        is_eval, same_crop_transform, multi_crop, crop_dim)

    img_transform = get_img_transform(is_eval, modality)

    return FrameReader(clip_dir, modality, crop_transform,
                       img_transform, same_crop_transform)


if __name__ == "__main__":
    # Training reader without crop
    reader = get_frame_reader(Path(
        "/data/wsf/projects/video-datasets/FineGym-frames"), is_eval=False, modality="rgb", same_crop_transform=False, multi_crop=False)
    clip = reader.load_frames("0jqn1vxdhls_E_000133_000140", 0, 308)
    print(clip.shape)

    # Training reader with crop
    reader = get_frame_reader(Path(
        "/data/wsf/projects/video-datasets/tennis-frames1"), is_eval=False, modality="rgb", same_crop_transform=False, multi_crop=False, crop_dim=100)
    clip = reader.load_frames(
        "usopen_2015_mens_final_federer_djokovic_10065_10383", 0, 318)
    print(clip.shape)

    # Testing reader without crop
    reader = get_frame_reader(Path(
        "/data/wsf/projects/video-datasets/fs_comp-frames"), is_eval=True, modality="rgb", same_crop_transform=False, multi_crop=False)
    clip = reader.load_frames(
        "men_olympic_short_program_2010_01_00011475_00015700", 0, 666)
    print(clip.shape)

    # Testing reader with crop
    reader = get_frame_reader(Path(
        "/data/wsf/projects/video-datasets/FineDiving-frames"), is_eval=True, modality="rgb", same_crop_transform=False, multi_crop=False, crop_dim=200)
    clip = reader.load_frames(
        "01__1", 0, 85)
    print(clip.shape)

    # Testing reader with multi crop
    reader = get_frame_reader(Path(
        "/data/wsf/projects/video-datasets/fs_perf-frames"), is_eval=True, modality="rgb", same_crop_transform=False, multi_crop=True, crop_dim=200)
    clip = reader.load_frames(
        "men_olympic_short_program_2010_01_00011475_00015700", 0, 4225)
    print(clip.shape)
