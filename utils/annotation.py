# Standard Library
from typing import TypedDict


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
