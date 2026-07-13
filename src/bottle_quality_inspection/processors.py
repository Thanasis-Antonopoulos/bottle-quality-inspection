"""Reusable frame processors."""

from dataclasses import dataclass

import numpy as np

from bottle_quality_inspection.roi import RegionOfInterest
from bottle_quality_inspection.video_io import VideoFrame


@dataclass(frozen=True, slots=True)
class RoiOverlayProcessor:
    """Draw one region of interest on every video frame."""

    region: RegionOfInterest
    color: tuple[int, int, int] = (0, 255, 255)
    thickness: int = 2
    show_label: bool = True

    def __call__(self, video_frame: VideoFrame) -> np.ndarray:
        """Return the video frame with the configured ROI drawn."""
        return self.region.draw(
            video_frame.image,
            color=self.color,
            thickness=self.thickness,
            show_label=self.show_label,
        )