# Development Guide

This document describes the development workflow for the modern Bottle Quality
Inspection application.

## Environment

The project currently targets:

```text
Python 3.12
OpenCV 4.x
NumPy 2.x
pytest
Ruff
```

Create the virtual environment:

```powershell
py -3.12 -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\Activate.ps1
```

Install the project in editable mode:

```powershell
python -m pip install -e ".[dev]"
```

Editable installation is important because the console command:

```text
bottle-inspection
```

is registered through `pyproject.toml`.

## Primary validation commands

Before committing code, run:

```powershell
ruff check .
pytest
```

Both must succeed.

The package CLI should also remain available:

```powershell
python -m bottle_quality_inspection --help
```

and:

```powershell
bottle-inspection --help
```

## Source layout

Production code belongs under:

```text
src/bottle_quality_inspection/
```

The current major areas are:

```text
video_io.py
    video reading, metadata, scanning and validated writing

roi.py
    validated rectangular regions of interest

pipeline.py
    generic frame-processing video pipeline

config.py
    TOML inspection configuration

tracking.py
    bottle passage segmentation and station association

bottle_result.py
    bottle-level temporal aggregation and final decisions

bottle_pipeline.py
    end-to-end physical-bottle inspection orchestration

reporting.py
    CSV and JSON output

visual_reporting.py
    annotated video and flagged-bottle images

inspections/
    individual inspection algorithms
```

## Inspection modules

The `inspections/` package contains the individual computer-vision checks.

Current modules include:

```text
foreign_object.py
liquid_level.py
cork.py
cork_presence.py
glass.py
```

Frame-level inspection code should remain separate from physical-bottle
aggregation.

For example:

```text
frame evidence
    ↓
bottle_result.py
    ↓
physical bottle decision
```

This separation makes it possible to inspect and test raw evidence without
changing the final decision policy.

## Historical compatibility

The project contains preserved historical implementations under:

```text
legacy/
```

These files are evidence of the original project and should not be modified as
part of modern development.

When reproducing historical behavior:

1. inspect the original implementation,
2. document what the original code actually did,
3. implement the behavior inside the modern package,
4. add automated tests,
5. compare results against the historical source.

Do not invent historical rules that were not present in the source.

## Modern decision policy

Not every modern rule is historical.

The current architecture distinguishes between:

```text
historical frame-level behavior
```

and:

```text
modern physical-bottle aggregation
```

Examples of modern logic include:

- multi-frame evidence aggregation
- cross-station bottle association
- GOOD / REVIEW / DEFECTIVE final decisions
- tracked cork-presence evaluation
- project-specific glass classification

Modern rules should be described as project policy rather than attributed to the
2022 implementation.

## Bottle-level status model

The final bottle status is based on the required inspection checks.

Priority is:

```text
DEFECTIVE
    ↓
REVIEW
    ↓
GOOD
```

If any check contains a confirmed defect:

```text
final status = DEFECTIVE
```

If there is no confirmed defect but at least one check requires review:

```text
final status = REVIEW
```

Otherwise:

```text
final status = GOOD
```

This logic is implemented in:

```text
src/bottle_quality_inspection/bottle_result.py
```

## Configuration

Inspection-specific parameters belong in TOML configuration files rather than
being scattered through CLI code.

The current historical reconstruction uses:

```text
configs/legacy-2022.toml
```

Configuration should be used for parameters such as:

- regions of interest
- thresholds
- template paths
- Hough parameters
- morphological settings
- inspection enable/disable flags

Historical profiles should remain historically accurate.

New experimental or production calibration should use separate configuration
profiles rather than silently changing historical values.

## Testing strategy

Tests are organized by component.

Examples:

```text
test_video_io.py
test_video_frames.py
test_video_writer.py
test_roi.py
test_config.py
test_tracking.py
test_foreign_object.py
test_liquid_level.py
test_cork.py
test_cork_presence.py
test_glass.py
test_bottle_result.py
test_reporting.py
test_visual_reporting.py
test_cli_inspect.py
```

Tests should normally avoid requiring private or ignored media files.

The original local inspection video is useful for manual end-to-end validation,
but the automated CI suite should be able to run from a clean Git checkout.

## Smoke testing

Smoke tests should verify that the installed application starts correctly
without requiring the private legacy video.

Typical examples include:

```powershell
python -m bottle_quality_inspection --help
```

```powershell
python -m bottle_quality_inspection inspect --help
```

```powershell
bottle-inspection --help
```

These checks verify package installation, imports, CLI registration, and parser
construction.

## Manual end-to-end validation

When the local historical video is available, run:

```powershell
bottle-inspection inspect `
    legacy\2022\media\20221014_155146-trim.mp4 `
    --config configs\legacy-2022.toml `
    --output-dir outputs\inspection `
    --overwrite
```

The currently validated historical-video result is:

```text
Associated bottles: 3

bottle-001 -> GOOD
bottle-002 -> REVIEW
bottle-003 -> GOOD

GOOD: 2
REVIEW: 1
DEFECTIVE: 0

Unmatched foreign passages: 0
Unmatched downstream passages: 1
```

These values should not be turned into universal product expectations. They are
the current regression result for the available historical recording.

## Diagnostic tools

Engineering utilities live under:

```text
tools/
```

See:

```text
tools/README.md
```

for their purpose and classification.

These scripts may depend on project-specific assets or local video files.

Production modules must not depend on the `tools/` directory.

## Outputs

Generated runtime artifacts belong under:

```text
outputs/
```

and should not be committed.

Typical inspection outputs include:

```text
inspection-results.csv
inspection-results.json
annotated-inspection.mp4
review/
rejected/
```

## Continuous integration

GitHub Actions runs:

```text
installation
Ruff
pytest
module CLI smoke check
installed CLI smoke check
```

The CI workflow is stored at:

```text
.github/workflows/ci.yml
```

The historical video is intentionally not required by CI.

## Branch workflow

Development should normally use short-lived feature branches.

Example:

```powershell
git checkout main
git pull origin main
git checkout -b feat/example
```

Before committing:

```powershell
ruff check .
pytest
git status --short
```

Then stage only the intended files.

Avoid blindly using:

```text
git add .
```

when generated files or unrelated work may be present.

## Pull requests

A pull request should explain:

- what changed
- why the change was needed
- how the change was tested
- any known limitations
- whether historical behavior was preserved or intentionally changed

CI should pass before merging into `main`.

## Design principle

The repository should remain understandable as both:

```text
a preserved Machine Vision project history
```

and:

```text
a modern production-oriented computer-vision application
```

Historical evidence should remain intact while modern code becomes cleaner,
more testable, more configurable, and easier to operate.