# Bottle Inspection Requirements

## Purpose

The system inspects bottles moving through a bottling line.

A bottle is accepted only when all configured required inspections pass.

The modern implementation must keep inspection parameters outside the Python
source code wherever practical so that a production setup can be calibrated
without modifying application logic.

The baseline configuration is:

`configs/legacy-2022.toml`

---

## Inspection sequence

The intended inspection pipeline is:

1. Bottle localization
2. Cork inspection
3. Liquid-level inspection
4. Foreign-object / dark-spot inspection
5. Glass-integrity inspection
6. Final bottle decision

The eventual production result must be associated with a physical bottle,
rather than treating every frame as an independent product.

---

## Bottle localization

### Legacy source

`bottle_template.jpg`

### Method

OpenCV normalized correlation template matching:

`TM_CCOEFF_NORMED`

### Legacy limitation

The 2022 video implementation selected the best bottle-template match but did
not define a minimum score for a bottle PASS/FAIL decision.

The modern implementation must therefore not invent a legacy bottle threshold.

---

## Cork inspection

### Legacy source

`correct_cork.jpg`

### Method

OpenCV normalized correlation template matching:

`TM_CCOEFF_NORMED`

### Legacy standalone threshold

`0.80`

The standalone 2022 cork experiment explicitly used an 80 percent threshold.

The video implementation located the best cork match but did not implement a
complete bottle-level PASS/FAIL decision.

---

## Foreign-object / dark-spot inspection

### Inspection ROI

Absolute video coordinates:

- x: 600
- y: 300
- width: 200
- height: 700

### Legacy preprocessing

- Gaussian kernel: 5
- Canny threshold 1: 17
- Canny threshold 2: 60
- Morphological closing kernel: 13
- Content blur kernel: 11
- Binary content threshold: 127
- Reflection threshold: 156
- Retained foreground components: 2

### Final decision region

The legacy final dark-pixel count was performed on:

- absolute x: 600 to 780
- absolute y: 570 to 850

Relative to the foreign-object ROI:

- x: 0
- y: 270
- width: 180
- height: 280

### Legacy rejection threshold

A bottle was flagged when the final count exceeded:

`900`

The modern implementation must reproduce and validate this behaviour before
introducing improved decision metrics.

---

## Liquid-level inspection

### Inspection ROI

Absolute video coordinates:

- x: 910
- y: 300
- width: 200
- height: 700

### Neck analysis region

Relative to the liquid-level ROI:

- x: 0
- y: 192
- width: 200
- height: 58

The 2022 source requested x=0:300, but the parent ROI is only 200 pixels wide,
so the effective crop width is 200 pixels.

### Legacy edge detection

The source passes these Canny arguments:

- threshold 1: 252
- threshold 2: 119

They are preserved exactly as written rather than silently reordered.

### Probabilistic Hough parameters

- rho: 1
- theta: 1 degree
- threshold: 15
- minimum line length: 20
- maximum line gap: 10

### Acceptable liquid level

The legacy code considers the result acceptable when:

`225 < level < 247`

---

## Glass integrity

### Legacy source

`broken_glass.jpg`

### Existing legacy work

The 2022 glass script experiments with:

- grayscale conversion
- average blur
- Gaussian blur
- Canny edge detection
- fixed binary thresholding
- inverted thresholding
- Otsu thresholding
- adaptive mean thresholding
- adaptive Gaussian thresholding

### Legacy limitation

The 2022 code does not define a completed glass-integrity PASS/FAIL algorithm.

Therefore the modern project must not invent a legacy decision rule.

Glass-integrity detection must be implemented and validated as a separate
inspection milestone.

---

## Final product decision

The target bottle result is conceptually:

| Check | Result |
| --- | --- |
| Bottle detected | PASS / FAIL |
| Cork | PASS / FAIL |
| Liquid level | PASS / FAIL |
| Foreign objects | PASS / FAIL |
| Glass integrity | PASS / FAIL |
| Final bottle result | PASS / FAIL |

When `require_all_checks = true`, any required inspection failure makes the
physical bottle defective.

The production implementation will later combine results across multiple
frames belonging to the same tracked bottle.