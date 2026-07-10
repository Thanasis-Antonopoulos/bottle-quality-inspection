# System Architecture

## Status

The architecture will be finalized after the original 2022 video-processing pipeline has been reviewed and reproduced.

## Planned processing flow

```text
Video input
    ↓
Frame preprocessing
    ↓
Bottle detection
    ↓
Bottle tracking
    ↓
Quality inspections
    ├── Cap or cork inspection
    ├── Fill-level inspection
    ├── Foreign-object inspection
    └── Bottle-damage inspection
    ↓
Bottle-level decision
    ↓
Annotated video and inspection report