# Standard Library
from typing import TypedDict

# Torch Library
import torch


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


class HierarchalClass(TypedDict):
    num_level: int
    num_trans: int
    level_dict: dict[int, dict[str, int]]
    trans_dict: dict[int, dict[str, str]]


class Example(TypedDict):
    frame: torch.FloatTensor
    level: int
    label: list[torch.FloatTensor]
    contains_event: int
