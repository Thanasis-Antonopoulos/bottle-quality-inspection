# 2022 Legacy Implementation Audit

## Purpose

This document records the behaviour and technical characteristics of the original 2022 Machine Vision implementation before its functionality is migrated into the modern production package.

The legacy files are preserved under:

```text
legacy/2022/
```

They will not be modified or imported directly by the new application.

## Reviewed files

```text
legacy/2022/scripts/
├── Glass preprocess.py
├── templatematch_cork.py
├── VIDEO_WITH_TEMPLATE_.py
└── VIDEO_WITH_TEMPLATE_AND_ROI.py
```

## Original processing flow

The implementation contains three main areas of experimentation:

1. Still-image preprocessing
2. Cork and bottle template matching
3. Video-based region-of-interest inspection

## Script responsibilities

### `Glass preprocess.py`

An exploratory still-image preprocessing script using `broken_glass.jpg`.

It demonstrates:

- Grayscale conversion
- Average and Gaussian blurring
- Canny edge detection
- Fixed binary thresholding
- Inverted thresholding
- Otsu thresholding
- Adaptive mean thresholding
- Adaptive Gaussian thresholding
- Histogram visualization

This script is primarily an experiment and is not part of the final video-processing pipeline.

### `templatematch_cork.py`

A still-image template-matching experiment.

It:

- Loads `felos.jpg`
- Loads `correct_cork.jpg` as a template
- Converts both images to grayscale
- Uses normalized correlation template matching
- Applies a match threshold of `0.80`
- Draws the matching region for visualization

### `VIDEO_WITH_TEMPLATE_.py`

A video-processing implementation based on template matching.

For every video frame, it:

- Converts the frame to grayscale
- Searches for the cork template
- Searches for the bottle template
- Selects the highest-scoring position for each template
- Draws bounding boxes
- Calculates processing FPS
- Displays the annotated frame
- Writes an annotated output video

This script performs frame-level template matching but does not assign persistent bottle IDs or produce one final decision per bottle.

### `VIDEO_WITH_TEMPLATE_AND_ROI.py`

The most complete 2022 inspection pipeline.

It defines two static inspection regions:

- A region for foreign-object or anomaly inspection
- A region for liquid-level inspection

The foreign-object workflow includes:

- Grayscale conversion
- Gaussian smoothing
- Canny edge detection
- Morphological closing
- Contour detection
- Bottle-area masking
- Connected-component analysis
- Edge filtering
- Black-pixel counting
- Defective-product warning annotation

The liquid-level workflow includes:

- Cropping the expected liquid-level area
- Canny edge detection
- Probabilistic Hough line detection
- Selecting the lowest detected horizontal line
- Comparing the detected level with fixed acceptable boundaries
- Annotating the frame as acceptable or unacceptable

The script also displays FPS and writes annotated frames to an output video.

## Confirmed strengths

The original implementation demonstrates:

- A real industrial inspection use case
- Full video reading and frame processing
- Region-of-interest inspection
- Template matching
- Morphological image processing
- Contour and connected-component analysis
- Hough line detection
- Visual defect warnings
- Annotated video output
- Basic runtime-performance measurement

These capabilities form the functional foundation of the modernized project.

## Confirmed technical limitations

### File and directory handling

- Input filenames are hardcoded as defaults.
- Template paths depend on the current working directory.
- The `outputs` directory is assumed to exist.
- Missing images and templates are not validated.
- Video writer initialization is not validated.

### Video validation

The scripts contain:

```python
if OSError:
    print("Program has stopped")
```

This does not test whether the video opened successfully. `OSError` is a class and always evaluates as true in this condition.

The correct implementation must check:

```python
capture.isOpened()
```

### Template matching

- The highest-scoring match is drawn even when confidence is poor.
- Match scores are not used to make a documented pass/fail decision.
- Template matching is repeated independently for every frame.
- No duplicate-match suppression is performed.
- No persistent bottle identity is maintained.
- One physical bottle can therefore be evaluated repeatedly.

### Static regions of interest

The inspection coordinates are hardcoded for one video resolution and camera position.

Examples include:

```text
x = 600–800
x = 910–1110
y = 300–1000
```

These coordinates will not automatically adapt to another camera, resolution, or production line.

### Hardcoded inspection thresholds

The implementation embeds parameters directly in source code, including:

- Canny thresholds
- Morphological kernel sizes
- Pixel-count thresholds
- Template-match thresholds
- Liquid-level boundaries
- Hough-line parameters
- Video output FPS

The modern implementation must move these values into validated configuration.

### Foreign-object inspection

- The function assumes at least one contour exists.
- Connected-component selection assumes sufficient components exist.
- Several intermediate values are printed for every frame.
- Pixel-by-pixel Python loops reduce performance.
- The defect threshold is based on one fixed black-pixel count.
- The result is a frame-level warning rather than a bottle-level result.

### Liquid-level inspection

- The inspection crop is fixed.
- The acceptable range is fixed.
- A plot may be opened from inside the video-processing loop.
- The selected line is based on the largest vertical coordinate.
- Detected lines are not filtered strongly by orientation.
- No confidence value is returned.
- Missing line detection is represented by a special numeric value rather than an explicit status.

### Runtime and output

- Processing, visualization, and business decisions are tightly coupled.
- `print()` statements are used instead of structured logging.
- Output metadata is not exported to CSV or JSON.
- There is no final inspection summary.
- The ROI video script does not explicitly release its video writer.
- There is no headless mode for server or edge deployment.

### Software structure

- Application logic executes immediately when files are imported.
- Functions depend on global files and hardcoded values.
- Similar processing logic is repeated.
- Unused variables and imports remain.
- There are no type hints.
- There are no automated tests.
- There is no configuration model.
- There is no error hierarchy.
- There is no separation between detection, visualization, reporting, and video I/O.

## Behaviour that must be reproduced first

The first modern baseline should reproduce these core capabilities:

1. Open the original input video safely.
2. Read every frame.
3. Apply the two original inspection regions.
4. Execute foreign-object inspection.
5. Execute liquid-level inspection.
6. Draw inspection annotations.
7. Calculate processing performance.
8. Save a valid annotated output video.
9. Release all video resources correctly.

The first baseline should not yet introduce:

- Machine-learning models
- A web interface
- Bottle tracking
- Crack detection
- Cloud deployment
- Major algorithm changes

These improvements should be added only after the original behaviour has been reproduced in a clean and testable structure.

## Planned migration order

```text
1. Configuration and path validation
2. Video reader and writer
3. Frame-level data models
4. Region-of-interest extraction
5. Foreign-object inspection
6. Liquid-level inspection
7. Frame annotation
8. Baseline command-line runner
9. Comparison with the 2022 output
10. Bottle tracking and bottle-level decisions
```

## Baseline success criteria

The modern baseline will be considered successful when it can:

- Process the original video from beginning to end
- Avoid unhandled exceptions
- Create a playable annotated output video
- Reproduce the main visual behaviour of the 2022 ROI pipeline
- Run without opening blocking Matplotlib windows
- Report the number of processed frames
- Report average processing FPS
- Pass automated tests