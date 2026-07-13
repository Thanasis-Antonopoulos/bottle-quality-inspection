# Bottle Quality Inspection

A production-oriented computer vision system for automated bottle quality inspection.

## Project status

This repository is an active modernization of a Machine Vision project originally developed in 2022.

The original implementation inspected bottles moving through an ouzo bottling process and detected quality problems using classical computer vision techniques.

The new implementation aims to provide a clean, configurable, tested, and deployment-ready inspection pipeline.

## Planned inspection capabilities

- Bottle detection
- Cap or cork inspection
- Liquid fill-level inspection
- Foreign-object detection
- Bottle damage and crack detection
- Annotated video generation
- Bottle-level inspection reports
- Performance and quality metrics

## Project objectives

The project will:

1. Reproduce the working behaviour of the original 2022 implementation.
2. Refactor the original scripts into reusable Python modules.
3. Add bottle tracking and bottle-level decisions.
4. Add automated tests and code-quality checks.
5. Provide configurable inspection rules.
6. Support deployment through a command-line application.
7. Provide a demonstration interface after the inspection engine is stable.

## Repository structure

```text
bottle-quality-inspection/
├── docs/                     Project documentation
├── legacy/                   Preserved historical implementations
│   ├── 2022/                 Original university implementation
│   └── 2025/                 Experimental revised implementation
├── src/
│   └── bottle_quality_inspection/
│                              Production application source code
├── tests/                    Automated tests
├── pyproject.toml            Python project configuration
└── README.md
```

## Development setup

Create and activate a virtual environment:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
```

Install the package and development tools:

```powershell
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Run the application:

```powershell
python -m bottle_quality_inspection
```

Run the tests:

```powershell
pytest
```

Run code-quality checks:

```powershell
ruff check .
```
## Video metadata

Inspect and validate a local video file:

```powershell
python -m bottle_quality_inspection metadata path\to\video.mp4
```

Example using the local legacy video:

```powershell
python -m bottle_quality_inspection metadata legacy\2022\media\20221014_155146-trim.mp4
```

The command reports:

- Video resolution
- Frames per second
- Frame count
- Estimated duration

It also validates that the file exists, is a regular file, can be opened by OpenCV, and contains valid dimensions.

## Video tools

### Read video metadata

Validate a local video and display its technical properties:

```powershell
python -m bottle_quality_inspection metadata path\to\video.mp4
```

Example output:

```text
Resolution: 2400 x 1080
FPS: 30.20
Frames: 92
Estimated duration: 3.05 seconds
```

The command validates that:

- The path exists
- The path points to a file
- OpenCV can open the video
- The video has valid dimensions

### Scan a complete video

Decode every video frame and report processing throughput:

```powershell
python -m bottle_quality_inspection scan path\to\video.mp4
```

Example using the original local test video:

```powershell
python -m bottle_quality_inspection scan legacy\2022\media\20221014_155146-trim.mp4
```

Example output:

```text
Frames decoded: 92
Elapsed time: 1.595 seconds
Processing FPS: 57.69
```

The scan command:

- Reads frames sequentially
- Assigns each frame an index
- Calculates frame timestamps
- Detects empty or invalid frames
- Releases OpenCV resources automatically
- Reports total decoding performance

The original video is excluded from Git and must be available locally.

### Copy and validate a video

Decode and re-encode a complete video into a validated output file:

```powershell
python -m bottle_quality_inspection copy `
    path\to\input.mp4 `
    outputs\copied-video.mp4
```

Replace an existing output file:

```powershell
python -m bottle_quality_inspection copy `
    path\to\input.mp4 `
    outputs\copied-video.mp4 `
    --overwrite
```

Use another four-character OpenCV codec:

```powershell
python -m bottle_quality_inspection copy `
    path\to\input.mp4 `
    outputs\copied-video.mp4 `
    --codec mp4v
```

The command validates:

- Input and output paths are different
- The codec contains exactly four characters
- The source video has valid dimensions and FPS
- Every decoded frame matches the expected dimensions
- The output writer opens successfully
- Existing outputs are not overwritten accidentally
- Incomplete output files are removed after failures

## Historical implementations

The original 2022 and experimental 2025 implementations will be preserved under `legacy/`.

The production application will not directly import code from the legacy folders. Relevant logic will be reviewed, tested, and migrated into the new package.

## Current version

```text
0.1.0 — Initial project structure
```