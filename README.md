# fundus-imaging-segmenter

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![NumPy](https://img.shields.io/badge/NumPy-latest-blue.svg)](https://numpy.org)
[![SciKit-Image](https://img.shields.io/badge/scikit--image-latest-blue.svg)](https://scikit-image.org)

> Retinal fundus image segmentation pipeline using NumPy and SciKit-Image. Automates detection of the optic disc, blood vessels, and lesions.

## Features

- **Optic Disc Detection** — Automated bright-region thresholding and centroid locating
- **Vessel Segmentation** — Local contrast enhancement and morphological filtering
- **Lesion Detection** — Exudate and drusen identification using adaptive thresholding
- **Performance Metrics** — Calculates Dice, Jaccard, sensitivity, and specificity
- **Batch Processing** — CLI support for single images or entire directories

## Tech Stack

- **Core**: Python 3.9+
- **Image Processing**: NumPy, SciKit-Image
- **Testing**: pytest (50+ cases)

## Project Structure

```
fundus-imaging-segmenter/
├── src/
│   └── main.py          # Image processing pipeline and CLI
├── tests/
│   └── test_segmenter.py # Suite of 52 unit tests
├── examples/             # Sample fundus images
└── README.md
```

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/DinhLucent/fundus-imaging-segmenter.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run segmentation:
   ```bash
   python src/main.py process examples/sample_eye.jpg
   ```

## Demo

Visualize detection steps using the debug flag:
```bash
python src/main.py process examples/sample_eye.jpg --debug
```

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---
Built by [DinhLucent](https://github.com/DinhLucent)
