"""Custom exceptions raised by the inspection application."""


class BottleInspectionError(Exception):
    """Base exception for Bottle Quality Inspection errors."""


class VideoOpenError(BottleInspectionError):
    """Raised when a video cannot be opened or contains invalid metadata."""


class VideoReadError(BottleInspectionError):
    """Raised when video frames cannot be decoded correctly."""


class VideoWriteError(BottleInspectionError):
    """Raised when an output video cannot be created or written."""


class RegionOfInterestError(BottleInspectionError):
    """Raised when a region of interest is invalid for an image frame."""

class ImageWriteError(BottleInspectionError):
    """Raised when an output image cannot be created or written."""

class FrameProcessingError(BottleInspectionError):
    """Raised when a frame processor returns an invalid result or fails."""