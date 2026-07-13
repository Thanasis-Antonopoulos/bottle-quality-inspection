"""Validated rectangular regions of interest for image frames."""

from dataclasses import dataclass

import cv2
import numpy as np

from bottle_quality_inspection.exceptions import RegionOfInterestError


@dataclass(frozen=True, slots=True)
class RegionOfInterest:
    """A named rectangular region inside an image frame."""

    name: str
    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        """Validate values that do not depend on a particular frame."""
        if not self.name.strip():
            raise RegionOfInterestError("Region name cannot be empty.")

        if self.x < 0 or self.y < 0:
            raise RegionOfInterestError(
                f"Region coordinates cannot be negative: x={self.x}, y={self.y}"
            )

        if self.width <= 0 or self.height <= 0:
            raise RegionOfInterestError(
                "Region dimensions must be greater than zero: "
                f"width={self.width}, height={self.height}"
            )

    @property
    def x_end(self) -> int:
        """Return the exclusive horizontal end coordinate."""
        return self.x + self.width

    @property
    def y_end(self) -> int:
        """Return the exclusive vertical end coordinate."""
        return self.y + self.height

    def validate_for_frame(self, frame: np.ndarray) -> None:
        """Validate that this region fits inside an image frame."""
        if not isinstance(frame, np.ndarray):
            raise TypeError("Frame must be a NumPy array.")

        if frame.size == 0:
            raise RegionOfInterestError("Frame cannot be empty.")

        if frame.ndim not in {2, 3}:
            raise RegionOfInterestError(
                f"Frame must have two or three dimensions, received {frame.ndim}."
            )

        frame_height, frame_width = frame.shape[:2]

        if self.x_end > frame_width or self.y_end > frame_height:
            raise RegionOfInterestError(
                f"Region '{self.name}' exceeds frame boundaries: "
                f"region=({self.x}, {self.y}, {self.width}, {self.height}), "
                f"frame={frame_width}x{frame_height}"
            )

    def extract(self, frame: np.ndarray) -> np.ndarray:
        """Extract and return an independent copy of this region."""
        self.validate_for_frame(frame)

        return frame[
            self.y : self.y_end,
            self.x : self.x_end,
        ].copy()

    def draw(
        self,
        frame: np.ndarray,
        *,
        color: tuple[int, int, int] = (0, 255, 255),
        thickness: int = 2,
        show_label: bool = True,
    ) -> np.ndarray:
        """Return a copy of the frame with the region drawn on it."""
        self.validate_for_frame(frame)

        if frame.ndim != 3 or frame.shape[2] != 3:
            raise RegionOfInterestError(
                "ROI drawing requires a three-channel BGR image."
            )

        if thickness <= 0:
            raise ValueError("Rectangle thickness must be greater than zero.")

        annotated = frame.copy()

        top_left = (self.x, self.y)
        bottom_right = (self.x_end - 1, self.y_end - 1)

        cv2.rectangle(
            annotated,
            top_left,
            bottom_right,
            color,
            thickness,
        )

        if show_label:
            label_y = max(self.y - 10, 20)

            cv2.putText(
                annotated,
                self.name,
                (self.x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                thickness,
                cv2.LINE_AA,
            )

        return annotated