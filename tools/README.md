# Engineering Tools

The scripts in this directory are development, calibration, diagnostic, and
validation utilities used while reconstructing and testing the Bottle Quality
Inspection system.

They are **not the primary application interface**.

For normal inspection use:

```powershell
bottle-inspection inspect `
    path\to\video.mp4 `
    --config configs\legacy-2022.toml `
    --output-dir outputs\inspection
```

or:

```powershell
python -m bottle_quality_inspection inspect `
    path\to\video.mp4 `
    --config configs\legacy-2022.toml `
    --output-dir outputs\inspection
```

## Tool categories

The tools are grouped conceptually into:

```text
diagnostics
calibration
validation
visualization
legacy development exporters
```

They remain as individual scripts because they were useful during reconstruction
and provide reproducible evidence for how the current implementation was
developed.

## Diagnostics

### `diagnose_liquid_level.py`

Examines the liquid-level detector frame by frame.

Use it when investigating:

- detected liquid levels
- Hough-line output
- transition-frame behavior
- acceptable-range decisions
- frames where no liquid level is detected

This tool was used to confirm the historical 2022 liquid-level behavior before
adding bottle-level temporal aggregation.

### `diagnose_cork.py`

Runs the historical cork-template matching approach across video frames.

It is useful for inspecting:

- template-match scores
- threshold behavior
- best-match positions
- instability of whole-frame cork matching

This tool helped demonstrate why whole-frame template matching was not sufficient
for reliable physical-bottle cork decisions.

### `diagnose_glass.py`

Examines glass preprocessing and feature measurements.

It is useful when reviewing:

- edge density
- horizontal-line evidence
- row-density evidence
- preprocessing output
- glass-classifier signals

The original 2022 source did not contain a completed glass PASS/FAIL rule, so
this tool supports the later project-specific glass implementation.

### `diagnose_bottle_passages.py`

Displays detected bottle passages at the configured inspection stations.

It is used to inspect:

- passage start frames
- passage end frames
- representative frames
- station sequence numbers
- upstream/downstream timing

This tool was important when moving from frame-level inspection to physical
bottle tracking.

### `diagnose_tracked_corks.py`

Evaluates cork-template evidence inside tracked bottle regions rather than over
the complete frame.

It is useful for comparing:

```text
cork search region
vs.
body control region
```

and for studying why location-specific evidence is more meaningful than
whole-frame template matching.

### `diagnose_multiframe_bottles.py`

Prints the frame-by-frame evidence belonging to each associated physical bottle.

It combines:

- foreign-object evidence
- liquid-level evidence
- glass evidence
- cork evidence
- tracking information

This diagnostic was used to design the current bottle-level temporal aggregation
policy.

## Calibration

### `calibrate_tracked_cork_presence.py`

Supports calibration of the modern multi-frame cork-presence logic.

It compares tracked cork-region scores with paired body-region control scores.

The available historical dataset contains visibly corked bottles but does not
contain a validated set of uncorked or physically defective cork examples.

For that reason, the current classifier supports:

```text
PRESENT
UNVERIFIED
```

rather than claiming reliable:

```text
PRESENT
ABSENT
```

classification.

### `sample_glass_video.py`

Extracts candidate bottle images from the video for glass-analysis work.

It was used while comparing normal bottle regions with the available broken
glass reference image.

## Validation

### `validate_glass_classifier.py`

Runs the calibrated project-specific glass classifier against the selected
reference and validation examples.

The resulting thresholds are specific to the current dataset and are not
universal manufacturing tolerances.

### `validate_tracked_cork_presence.py`

Validates the multi-frame cork-presence logic on associated bottle passages.

It reports:

- selected frames
- top-k cork score median
- paired body-control median
- score margin
- final presence status

## Comparison and visualization

### `compare_glass_reference.py`

Compares glass features between available reference images.

This is primarily an engineering-analysis utility rather than a production
inspection command.

### `visualize_bottle_associations.py`

Visualizes the relationship between upstream and downstream bottle passages.

Use it when investigating:

- bottle passage pairing
- temporal offsets
- unmatched passages
- station overlap

## Legacy development exporters

The following scripts were created before reporting was integrated into the main
application CLI.

### `inspect_tracked_bottles.py`

Runs the physical-bottle inspection pipeline and prints detailed results to the
terminal.

Its functionality is now largely represented by:

```powershell
bottle-inspection inspect ...
```

It is retained because its verbose output remains useful during development.

### `export_inspection_reports.py`

Exports CSV and JSON bottle reports.

This capability is now included automatically in:

```powershell
bottle-inspection inspect ...
```

The script remains available as a focused development utility.

### `export_visual_inspection.py`

Exports the annotated video and flagged bottle images.

This capability is also now included automatically in:

```powershell
bottle-inspection inspect ...
```

The script remains useful when visual artifacts need to be generated
independently during development.

## Production interface vs engineering tools

The supported product interface is:

```text
bottle-inspection
```

The scripts in `tools/` should be treated as:

```text
engineering support utilities
```

rather than stable public APIs.

New production functionality should normally be implemented under:

```text
src/bottle_quality_inspection/
```

and exposed through the main CLI when appropriate.

Diagnostic or calibration experiments may remain under:

```text
tools/
```

provided that they do not become dependencies of the production package.

## Legacy source rule

Scripts under:

```text
legacy/
```

are historical artifacts.

They should not be modified to implement modern functionality.

When historical behavior is needed, it should be:

1. reviewed,
2. documented,
3. reconstructed inside `src/bottle_quality_inspection/`,
4. tested,
5. compared against the historical behavior.

This keeps the legacy implementation intact while allowing the modern
application to evolve safely.