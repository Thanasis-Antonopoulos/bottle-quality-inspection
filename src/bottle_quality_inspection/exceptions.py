"""Custom exceptions raised by the inspection application."""


class BottleInspectionError(Exception):
    """Base exception for Bottle Quality Inspection errors."""


class VideoOpenError(BottleInspectionError):
    """Raised when a video cannot be opened or contains invalid metadata."""


class VideoReadError(BottleInspectionError):
    """Raised when video frames cannot be decoded correctly."""