"""Tests for the reusable frame-processing pipeline."""

from pathlib import Path

import numpy as np
import pytest

from bottle_quality_inspection import pipeline
from bottle_quality_inspection.exceptions import (
    FrameProcessingError,
    VideoWriteError,
)
from bottle_quality_inspection.video_io import VideoFrame, VideoMetadata


class FakeWriter:
    """Configurable OpenCV video writer used by pipeline tests."""

    def __init__(self, *, opened: bool = True) -> None:
        self.opened = opened
        self.frames: list[np.ndarray] = []
        self.released = False

    def isOpened(self) -> bool:
        """Return the configured writer state."""
        return self.opened

    def write(self, frame: np.ndarray) -> None:
        """Store a copy of a processed frame."""
        self.frames.append(frame.copy())

    def release(self) -> None:
        """Record that the writer resource was released."""
        self.released = True


def create_metadata(source: Path) -> VideoMetadata:
    """Create consistent metadata for a test video."""
    return VideoMetadata(
        path=source.resolve(),
        width=640,
        height=480,
        fps=25.0,
        frame_count=3,
    )


def create_frames() -> list[VideoFrame]:
    """Create three valid test video frames."""
    return [
        VideoFrame(
            index=0,
            timestamp_seconds=0.00,
            image=np.zeros((480, 640, 3), dtype=np.uint8),
        ),
        VideoFrame(
            index=1,
            timestamp_seconds=0.04,
            image=np.ones((480, 640, 3), dtype=np.uint8),
        ),
        VideoFrame(
            index=2,
            timestamp_seconds=0.08,
            image=np.full((480, 640, 3), 2, dtype=np.uint8),
        ),
    ]


def configure_video_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    source: Path,
    fake_writer: FakeWriter,
) -> None:
    """Replace video input and output dependencies with test doubles."""
    monkeypatch.setattr(
        pipeline,
        "read_video_metadata",
        lambda _: create_metadata(source),
    )
    monkeypatch.setattr(
        pipeline,
        "iter_video_frames",
        lambda _: iter(create_frames()),
    )
    monkeypatch.setattr(
        pipeline.cv2,
        "VideoWriter_fourcc",
        lambda *args: 1234,
    )
    monkeypatch.setattr(
        pipeline.cv2,
        "VideoWriter",
        lambda *args: fake_writer,
    )


def test_process_video_processes_and_writes_every_frame(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Every source frame should be processed and written once."""
    source = tmp_path / "source.mp4"
    destination = tmp_path / "processed.mp4"
    fake_writer = FakeWriter()

    configure_video_dependencies(
        monkeypatch,
        source,
        fake_writer,
    )

    processed_indexes: list[int] = []

    def processor(video_frame: VideoFrame) -> np.ndarray:
        processed_indexes.append(video_frame.index)
        return video_frame.image + 10

    result = pipeline.process_video(
        source,
        destination,
        processor,
    )

    assert result.input_path == source.resolve()
    assert result.output_path == destination.resolve()
    assert result.frames_processed == 3
    assert result.codec == "mp4v"
    assert result.processing_fps is not None
    assert result.processing_fps > 0

    assert processed_indexes == [0, 1, 2]
    assert len(fake_writer.frames) == 3
    assert np.all(fake_writer.frames[0] == 10)
    assert np.all(fake_writer.frames[1] == 11)
    assert np.all(fake_writer.frames[2] == 12)
    assert fake_writer.released is True


def test_process_video_rejects_invalid_processed_dimensions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A processor must preserve the source video dimensions."""
    source = tmp_path / "source.mp4"
    destination = tmp_path / "processed.mp4"
    fake_writer = FakeWriter()

    configure_video_dependencies(
        monkeypatch,
        source,
        fake_writer,
    )

    def processor(_: VideoFrame) -> np.ndarray:
        return np.zeros((100, 100, 3), dtype=np.uint8)

    with pytest.raises(
        FrameProcessingError,
        match="dimensions differ",
    ):
        pipeline.process_video(
            source,
            destination,
            processor,
        )

    assert fake_writer.released is True


def test_process_video_wraps_processor_exception(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Unexpected processor failures should include frame context."""
    source = tmp_path / "source.mp4"
    destination = tmp_path / "processed.mp4"
    fake_writer = FakeWriter()

    configure_video_dependencies(
        monkeypatch,
        source,
        fake_writer,
    )

    def processor(_: VideoFrame) -> np.ndarray:
        raise RuntimeError("Unexpected processing failure")

    with pytest.raises(
        FrameProcessingError,
        match="Frame processor failed",
    ):
        pipeline.process_video(
            source,
            destination,
            processor,
        )

    assert fake_writer.released is True


def test_process_video_rejects_existing_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Existing output files should be protected by default."""
    source = tmp_path / "source.mp4"
    destination = tmp_path / "processed.mp4"
    destination.write_bytes(b"existing")

    monkeypatch.setattr(
        pipeline,
        "read_video_metadata",
        lambda _: create_metadata(source),
    )

    with pytest.raises(
        FileExistsError,
        match="already exists",
    ):
        pipeline.process_video(
            source,
            destination,
            lambda frame: frame.image,
        )


def test_process_video_rejects_writer_that_cannot_open(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A failed OpenCV writer should raise a clear error."""
    source = tmp_path / "source.mp4"
    destination = tmp_path / "processed.mp4"
    fake_writer = FakeWriter(opened=False)

    configure_video_dependencies(
        monkeypatch,
        source,
        fake_writer,
    )

    with pytest.raises(
        VideoWriteError,
        match="could not create",
    ):
        pipeline.process_video(
            source,
            destination,
            lambda frame: frame.image,
        )

    assert fake_writer.released is True