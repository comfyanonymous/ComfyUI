import logging
from typing import Optional

import torch
from comfy_api.latest import Input


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


def validate_image_aspect_ratio_range(
    image: torch.Tensor,
    min_ratio: tuple[float, float],  # e.g. (1, 4)
    max_ratio: tuple[float, float],  # e.g. (4, 1)
    *,
    strict: bool = True,             # True -> (min, max); False -> [min, max]
) -> float:
    a1, b1 = min_ratio
    a2, b2 = max_ratio
    if a1 <= 0 or b1 <= 0 or a2 <= 0 or b2 <= 0:
        raise ValueError("Ratios must be positive, like (1, 4) or (4, 1).")
    lo, hi = (a1 / b1), (a2 / b2)
    if lo > hi:
        lo, hi = hi, lo
        a1, b1, a2, b2 = a2, b2, a1, b1  # swap only for error text
    w, h = get_image_dimensions(image)
    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid image dimensions: {w}x{h}")
    ar = w / h
    ok = (lo < ar < hi) if strict else (lo <= ar <= hi)
    if not ok:
        op = "<" if strict else "≤"
        raise ValueError(f"Image aspect ratio {ar:.6g} is outside allowed range: {a1}:{b1} {op} ratio {op} {a2}:{b2}")
    return ar


def validate_aspect_ratio_closeness(
    start_img,
    end_img,
    min_rel: float,
    max_rel: float,
    *,
    strict: bool = False,   # True => exclusive, False => inclusive
) -> None:
    w1, h1 = get_image_dimensions(start_img)
    w2, h2 = get_image_dimensions(end_img)
    if min(w1, h1, w2, h2) <= 0:
        raise ValueError("Invalid image dimensions")
    ar1 = w1 / h1
    ar2 = w2 / h2
    # Normalize so it is symmetric (no need to check both ar1/ar2 and ar2/ar1)
    closeness = max(ar1, ar2) / min(ar1, ar2)
    limit = max(max_rel, 1.0 / min_rel)  # for 0.8..1.25 this is 1.25
    if (closeness >= limit) if strict else (closeness > limit):
        raise ValueError(f"Aspect ratios must be close: start/end={ar1/ar2:.4f}, allowed range {min_rel}–{max_rel}.")


def validate_video_dimensions(
    video: Input.Video,
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
    video: Input.Video,
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


def get_number_of_images(images):
    if isinstance(images, torch.Tensor):
        return images.shape[0] if images.ndim >= 4 else 1
    return len(images)


def validate_audio_duration(
    audio: Input.Audio,
    min_duration: Optional[float] = None,
    max_duration: Optional[float] = None,
) -> None:
    sr = int(audio["sample_rate"])
    dur = int(audio["waveform"].shape[-1]) / sr
    eps = 1.0 / sr
    if min_duration is not None and dur + eps < min_duration:
        raise ValueError(f"Audio duration must be at least {min_duration}s, got {dur + eps:.2f}s")
    if max_duration is not None and dur - eps > max_duration:
        raise ValueError(f"Audio duration must be at most {max_duration}s, got {dur - eps:.2f}s")
