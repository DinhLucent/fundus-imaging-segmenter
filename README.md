# fundus-imaging-segmenter

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![NumPy](https://img.shields.io/badge/NumPy-1.20+-blue?logo=numpy)
![License](https://img.shields.io/badge/License-MIT-green)

A high-performance, pure-NumPy toolkit for anatomical segmentation and pathology detection in retinal fundus images. It provides reliable screening aids without the overhead of deep learning frameworks.

## What is Fundus Imaging?

Retinal fundus photography involves capturing a 2D representation of the inner surface of the eye. It is the primary tool for screening eye diseases like Diabetic Retinopathy, Glaucoma, and Macular Degeneration.

This tool segments three critical features:
1. **Optic Disc**: The entry point for the optic nerve. Its size and shape are key for Glaucoma screening.
2. **Blood Vessels**: Used to detect hypertensive retinopathy and vessel branching anomalies.
3. **Lesions**: Detects bright pathologies like exudates or drusen, which indicate active disease processes.

## Quick Start

### Segment an image

Load a NumPy image file (`.npy`) and run the full pipeline:

```bash
python -m src.main segment image.npy --summary
```

Output:
```
══════════════════════════════════════════════════
  FUNDUS SCREENING CLINICAL SUMMARY
══════════════════════════════════════════════════
  IMAGE DATA
  ├─ Resolution : 512 x 512
  └─ Status     : PROCESSED

  ANATOMY
  ├─ Optic Disc : FOUND
  │  └─ Area    : 4.21% of retina
  └─ Vasculature: 12.45% density

  PATHOLOGY
  ├─ Lesions    : 3
  └─ Risk Level : MONITOR
══════════════════════════════════════════════════
```

## Features

- **Classical CV Core**: Uses Otsu thresholding, morphological operations, and connected components.
- **Vectorized Performance**: Convolution-like smoothing and background estimation are fully vectorized using NumPy strides.
- **Dependency-Lite**: Requires only NumPy. No OpenCV or heavy Deep Learning weights needed.
- **Rich Metrics**: Calculates Dice coefficient, Jaccard index, Sensitivity, and Specificity for evaluation.
- **Clinical Reporting**: Built-in summary formatter for human-readable risk assessment.

## How it works — module by module

### `src/main.py` — Segmentation Engine

Contains the dedicated segmenter classes and the main orchestrator.

#### Anatomical Segmentation

```python
import numpy as np
from src.main import FundusImagingSegmenter, SegmentationTarget

# Initialize the pipeline
segmenter = FundusImagingSegmenter()

# Load image (H, W, 3 uint8)
image = np.load("my_retina.npy")

# Run segmentation for all targets
result = segmenter.segment(image, target=SegmentationTarget.ALL)

print(f"Optic Disc Center: {result.optic_disc_bbox.center}")
print(f"Vessel Density: {result.metrics['vessel_density']:.2%}")
```

#### Pathology Detection

The `LesionSegmenter` uses local background estimation to find "bright" residuals which represent potential exudates.

```python
from src.main import LesionSegmenter

lesion_seg = LesionSegmenter(min_lesion_size=10)
mask, count = lesion_seg.segment(image, disc_mask=result.optic_disc_mask)

print(f"Detected {count} potential lesions.")
```

## Project Structure

```
fundus-imaging-segmenter/
├── src/
│   ├── __init__.py
│   └── main.py             # Core segmenters, metrics, and CLI
├── tests/
│   ├── test_segmenter.py   # Comprehensive tests for all CV algorithms
│   └── test_placeholder.py
├── docs/                   # Documentation assets
├── examples/               # Example notebooks/scripts
├── requirements.txt
├── LICENSE
└── README.md
```

## Installation

```bash
git clone https://github.com/DinhLucent/fundus-imaging-segmenter.git
cd fundus-imaging-segmenter
pip install -r requirements.txt
```

## Running Tests

The test suite covers algorithmic correctness, edge cases (black images, uniform images), and CLI integration.

```bash
# Run all tests
python -m pytest tests/test_segmenter.py -v
```

## License

MIT License — see [LICENSE](LICENSE)

---
Built by [DinhLucent](https://github.com/DinhLucent)
