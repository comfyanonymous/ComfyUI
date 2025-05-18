import logging
from typing import Optional

import torch
from comfy_api.input.video_types import VideoInput


def get_image_dimensions(image: torch.Tensor) -> tuple[int, int]:
    if len(image.shape) == 4:
        return image.shape[1], image.shape[2]
    elif len(image.shape) == 3:
        return image.shape[0], image.shape[1]
    else:
        raise ValueError("Invalid image tensor shape.")


def validate_image_dimensions(
    image: torch.Tensor,
    min_width: Optional[int] = None,
    max_width: Optional[int] = None,
    min_height: Optional[int] = None,
    max_height: Optional[int] = None,
):
    height, width = get_image_dimensions(image)

    if min_width is not None and width < min_width:
        raise ValueError(f"Image width must be at least {min_width}px, got {width}px")
    if max_width is not None and width > max_width:
        raise ValueError(f"Image width must be at most {max_width}px, got {width}px")
    if min_height is not None and height < min_height:
        raise ValueError(
            f"Image height must be at least {min_height}px, got {height}px"
        )
    if max_height is not None and height > max_height:
        raise ValueError(f"Image height must be at most {max_height}px, got {height}px")


def validate_image_aspect_ratio(
    image: torch.Tensor,
    min_aspect_ratio: Optional[float] = None,
    max_aspect_ratio: Optional[float] = None,
):
    width, height = get_image_dimensions(image)
    aspect_ratio = width / height

    if min_aspect_ratio is not None and aspect_ratio < min_aspect_ratio:
        raise ValueError(
            f"Image aspect ratio must be at least {min_aspect_ratio}, got {aspect_ratio}"
        )
    if max_aspect_ratio is not None and aspect_ratio > max_aspect_ratio:
        raise ValueError(
            f"Image aspect ratio must be at most {max_aspect_ratio}, got {aspect_ratio}"
        )


def validate_video_dimensions(
    video: VideoInput,
    min_width: Optional[int] = None,
    max_width: Optional[int] = None,
    min_height: Optional[int] = None,
    max_height: Optional[int] = None,
):
    try:
        width, height = video.get_dimensions()
    except Exception as e:
        logging.error("Error getting dimensions of video: %s", e)
        return

    if min_width is not None and width < min_width:
        raise ValueError(f"Video width must be at least {min_width}px, got {width}px")
    if max_width is not None and width > max_width:
        raise ValueError(f"Video width must be at most {max_width}px, got {width}px")
    if min_height is not None and height < min_height:
        raise ValueError(
            f"Video height must be at least {min_height}px, got {height}px"
        )
    if max_height is not None and height > max_height:
        raise ValueError(f"Video height must be at most {max_height}px, got {height}px")


def validate_video_duration(
    video: VideoInput,
    min_duration: Optional[float] = None,
    max_duration: Optional[float] = None,
):
    try:
        duration = video.get_duration()
    except Exception as e:
        logging.error("Error getting duration of video: %s", e)
        return

    epsilon = 0.0001
    if min_duration is not None and min_duration - epsilon > duration:
        raise ValueError(
            f"Video duration must be at least {min_duration}s, got {duration}s"
        )
    if max_duration is not None and duration > max_duration + epsilon:
        raise ValueError(
            f"Video duration must be at most {max_duration}s, got {duration}s"
        )
