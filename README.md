# Bottle Quality Inspection

A production-oriented computer vision system for automated bottle quality inspection.

This repository modernizes a Machine Vision project originally developed in 2022 for an ouzo bottling process. The original experimental scripts are preserved for historical reference, while the modern implementation provides reusable inspection modules, configuration, bottle tracking, multi-frame decisions, automated tests, structured reports, and a command-line application.

## Current status

The project now supports an end-to-end bottle inspection workflow.

A source video can be processed into physical bottle-level decisions using multiple inspection stations and multiple frames per bottle.

Current checks include:

- Foreign-object / dark-spot inspection
- Liquid fill-level inspection
- Glass-integrity inspection
- Cork-presence inspection
- Bottle passage detection
- Cross-station bottle association
- Multi-frame evidence aggregation
- Final GOOD / REVIEW / DEFECTIVE bottle decisions
- CSV reporting
- JSON reporting
- Annotated inspection video
- Review evidence images
- Rejected bottle evidence images

The original 2022 computer-vision behavior is preserved where a historical decision rule existed. Newer aggregation and glass-classification rules are explicitly treated as modern project logic rather than rewritten history.

## Inspection workflow

```text
Input video
    |
    v
Bottle passage detection
    |
    +--------------------------+
    |                          |
    v                          v
Foreign-object station     Downstream station
                               |
                               +--> Liquid level
                               +--> Glass integrity
                               +--> Cork presence
    |                          |
    +------------+-------------+
                 |
                 v
      Cross-station association
                 |
                 v
       Multi-frame aggregation
                 |
                 v
        Physical bottle result
                 |
        +--------+--------+
        |        |        |
       GOOD    REVIEW  DEFECTIVE
                 |
                 v
    CSV / JSON / annotated video
```

## Final bottle decisions

The final decision belongs to a tracked physical bottle rather than to an individual video frame.

`GOOD` means that every required bottle-level check passed.

`DEFECTIVE` means that at least one inspection check produced sufficiently strong repeated evidence of a confirmed defect.

`REVIEW` means that the system found uncertain or inconsistent evidence that should not automatically be treated as either good or defective.

This distinction is important because individual video frames can contain transition noise, partial visibility, reflections, or temporary occlusion.

## Example result

Using the original 2022 inspection video, the current pipeline identifies three complete physical bottle passages:

```text
bottle-001 -> GOOD
bottle-002 -> REVIEW
bottle-003 -> GOOD
```

The REVIEW decision for `bottle-002` is caused by an isolated foreign-object threshold crossing that was not repeated strongly enough to confirm a defect.

The recording also contains one unmatched downstream bottle because the video begins after that bottle has already passed the upstream inspection station.

## Installation

Python 3.12 is currently required.

Create and activate a virtual environment:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
```

Upgrade pip:

```powershell
python -m pip install --upgrade pip
```

Install the project in editable mode together with development tools:

```powershell
python -m pip install -e ".[dev]"
```

## Command-line application

After installation, the project can be run either as a Python module:

```powershell
python -m bottle_quality_inspection --help
```

or through the installed console command:

```powershell
bottle-inspection --help
```

### Complete inspection

Run the full physical-bottle inspection workflow:

```powershell
bottle-inspection inspect `
    path\to\input.mp4 `
    --config configs\legacy-2022.toml `
    --output-dir outputs\inspection
```

Replace existing output artifacts:

```powershell
bottle-inspection inspect `
    path\to\input.mp4 `
    --config configs\legacy-2022.toml `
    --output-dir outputs\inspection `
    --overwrite
```

The equivalent module command is:

```powershell
python -m bottle_quality_inspection inspect `
    path\to\input.mp4 `
    --config configs\legacy-2022.toml `
    --output-dir outputs\inspection `
    --overwrite
```

## Inspection outputs

A complete inspection produces:

```text
outputs/inspection/
├── inspection-results.csv
├── inspection-results.json
├── annotated-inspection.mp4
├── review/
│   └── bottle-XXX-frame-YYY.jpg
└── rejected/
    └── bottle-XXX-frame-YYY.jpg
```

### CSV

The CSV report contains one row per associated physical bottle.

It includes fields such as:

```text
bottle_id
final_status
foreign_object_status
foreign_defective_frames
foreign_max_black_pixels
liquid_level_status
liquid_median_level
glass_status
glass_defective_frames
cork_status
cork_top_k_median
cork_margin
foreign_start_frame
foreign_end_frame
downstream_start_frame
downstream_end_frame
travel_frame_offset
defective_checks
review_checks
```

### JSON

The JSON report preserves the richer nested inspection information, including:

```text
run summary
physical bottle IDs
tracking information
per-check evidence
liquid measurements
foreign-object measurements
glass results
cork-presence evidence
review reasons
confirmed defect reasons
unmatched station passages
```

### Annotated video

The annotated video displays the active physical bottle ID, inspection station, and final bottle status while the bottle passes through the configured inspection regions.

### Review and rejected images

Bottles with uncertain evidence are written to:

```text
review/
```

Confirmed defective bottles are written to:

```text
rejected/
```

GOOD bottles do not produce flagged evidence images.

## Inspection configuration

Inspection parameters are stored in TOML profiles.

The historical reconstruction is currently configured through:

```text
configs/legacy-2022.toml
```

This profile contains the original inspection regions and historical thresholds where they could be recovered from the 2022 source code.

Configuration includes:

- Foreign-object ROI
- Foreign-object decision region
- Canny thresholds
- Morphological parameters
- Historical foreign-object pixel threshold
- Liquid-level ROI
- Liquid-level neck region
- Hough line parameters
- Acceptable liquid-level range
- Cork template
- Historical cork template threshold
- Glass reference image

## Inspection components

### Foreign objects

The foreign-object detector reconstructs the original 2022 image-processing pipeline.

The historical frame-level defect rule is preserved:

```text
decision black pixels > 900
```

The modern bottle-level layer evaluates evidence across the complete upstream bottle passage instead of using only one frame.

### Liquid level

The liquid-level implementation preserves the historical edge detection and Hough-line logic.

The original acceptable range is:

```text
225 < liquid level < 247
```

The modern bottle-level decision uses the median of valid multi-frame measurements so that entry and exit transition frames do not dominate the physical-bottle result.

### Glass integrity

The original project contained several glass preprocessing experiments but no complete historical PASS/FAIL classifier.

The modern implementation therefore separates:

```text
historical preprocessing
```

from:

```text
2026 project-specific glass classification
```

The current glass thresholds are calibrated to the available project images and video and should not be interpreted as universal manufacturing tolerances.

### Cork presence

The historical project used grayscale template matching with:

```text
TM_CCOEFF_NORMED
threshold = 0.80
```

Whole-frame matching was found to be unreliable when multiple bottles were visible simultaneously.

The modern implementation instead evaluates cork evidence inside the tracked bottle ROI across multiple frames.

Current cork classification confirms:

```text
PRESENT
```

or:

```text
UNVERIFIED
```

It does not claim to identify physical cork damage because the available legacy dataset does not contain labeled defective-cork examples.

## Bottle tracking

The current tracking model is designed for the fixed-camera conveyor sequence present in the original project.

It detects bottle passages independently at the upstream and downstream inspection stations and associates them using monotonic conveyor order and temporal offset.

This is intentionally simpler than a generic object re-identification system.

## Repository structure

```text
bottle-quality-inspection/
├── configs/
│   └── legacy-2022.toml
├── docs/
│   └── inspection-requirements.md
├── legacy/
│   ├── 2022/
│   │   ├── assets/
│   │   ├── media/
│   │   └── scripts/
│   └── 2025/
├── src/
│   └── bottle_quality_inspection/
│       ├── inspections/
│       ├── bottle_pipeline.py
│       ├── bottle_result.py
│       ├── config.py
│       ├── reporting.py
│       ├── tracking.py
│       ├── video_io.py
│       └── visual_reporting.py
├── tests/
├── tools/
├── pyproject.toml
└── README.md
```

## Legacy implementations

Historical work is preserved under `legacy/`.

The production application does not directly import the historical scripts.

Relevant algorithms were reviewed and then reconstructed inside the modern package with validation tests.

The local original video:

```text
legacy/2022/media/20221014_155146-trim.mp4
```

is intentionally excluded from Git.

## Diagnostic tools

The `tools/` directory contains engineering utilities used during reconstruction, calibration, and validation.

These scripts include diagnostics for:

- liquid-level detection
- cork matching
- tracked cork evidence
- glass preprocessing
- glass classifier validation
- bottle passage detection
- bottle association visualization
- multi-frame bottle evidence
- report export
- visual inspection export

They support development and calibration but are not the primary product interface.

The supported application interface is:

```powershell
bottle-inspection
```

or:

```powershell
python -m bottle_quality_inspection
```

## Development

Run the complete automated test suite:

```powershell
pytest
```

Run code-quality validation:

```powershell
ruff check .
```

Run both before committing:

```powershell
ruff check .
pytest
```

## Current limitations

The project currently has several intentional limitations:

- Cork inspection confirms presence, not physical cork quality.
- No labeled uncorked or physically defective cork dataset is currently available.
- Glass decision thresholds are project-specific rather than universal.
- Bottle association assumes fixed inspection stations and monotonic conveyor motion.
- The current configuration is calibrated to the historical project footage.
- The project is not yet validated against a large industrial dataset.
- The inspection engine currently processes recorded video rather than a production PLC-triggered live camera stream.

These limitations are kept explicit so that the system does not claim capabilities unsupported by the available data.

## Roadmap

Planned next steps include:

1. Larger labeled datasets for each defect type.
2. Better bottle localization and tracking for more varied scenes.
3. True cork-defect examples and classification.
4. Industrial glass-damage validation.
5. Performance benchmarking on longer videos.
6. Live-camera inspection support.
7. Production logging and monitoring.
8. Containerized deployment.
9. Demonstration interface.
10. Integration with rejection hardware or manufacturing systems.

## Version

```text
0.1.0
```

The project is under active development.