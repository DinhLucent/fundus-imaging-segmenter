"""
fundus-imaging-segmenter — Python tool for retinal fundus image segmentation:
optic disc, blood vessels, and lesion detection using classical CV and scikit-image.

Author: DinhLucent
License: MIT

This module provides pure-Python / scikit-image based segmentation methods
that work on standard 3-channel (RGB) fundus images without requiring deep
learning frameworks.
"""
from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Domain types
# ---------------------------------------------------------------------------

class SegmentationTarget(Enum):
    OPTIC_DISC = "optic_disc"
    BLOOD_VESSELS = "blood_vessels"
    LESIONS = "lesions"
    ALL = "all"


@dataclass
class BoundingBox:
    x: int
    y: int
    w: int
    h: int

    @property
    def center(self) -> Tuple[int, int]:
        return (self.x + self.w // 2, self.y + self.h // 2)

    @property
    def area(self) -> int:
        return self.w * self.h

    def to_dict(self) -> Dict[str, int]:
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h}

    def __repr__(self) -> str:
        return f"BoundingBox(x={self.x}, y={self.y}, w={self.w}, h={self.h}, area={self.area})"



@dataclass
class SegmentationResult:
    """Holds all segmentation outputs for a fundus image."""
    image_shape: Tuple[int, int, int]
    optic_disc_mask: Optional[np.ndarray] = None
    optic_disc_bbox: Optional[BoundingBox] = None
    vessel_mask: Optional[np.ndarray] = None
    lesion_mask: Optional[np.ndarray] = None
    lesion_count: int = 0
    metrics: Dict[str, float] = field(default_factory=dict)

    @property
    def height(self) -> int:
        return self.image_shape[0]

    @property
    def width(self) -> int:
        return self.image_shape[1]

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "image_shape": list(self.image_shape),
            "optic_disc_bbox": self.optic_disc_bbox.to_dict() if self.optic_disc_bbox else None,
            "has_vessel_mask": self.vessel_mask is not None,
            "has_lesion_mask": self.lesion_mask is not None,
            "lesion_count": self.lesion_count,
            "metrics": self.metrics,
        }
        return d

    def __repr__(self) -> str:
        return f"SegmentationResult(shape={self.image_shape}, lesions={self.lesion_count}, has_od={self.optic_disc_bbox is not None})"



# ---------------------------------------------------------------------------
# Image utilities (pure NumPy — no skimage required for core logic)
# ---------------------------------------------------------------------------

def _rgb_to_gray(image: np.ndarray) -> np.ndarray:
    """Convert RGB uint8 image to float64 grayscale via ITU-R 601 luminance."""
    if image.ndim == 2:
        return image.astype(np.float64) / 255.0
    return (
        0.299 * image[:, :, 0] +
        0.587 * image[:, :, 1] +
        0.114 * image[:, :, 2]
    ).astype(np.float64) / 255.0


def _normalize(arr: np.ndarray) -> np.ndarray:
    """Min-max normalize array to [0, 1]."""
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-10:
        return np.zeros_like(arr, dtype=np.float64)
    return (arr - mn) / (mx - mn)


def _threshold_otsu(image: np.ndarray) -> float:
    """Compute Otsu threshold on a float [0,1] grayscale image."""
    hist, bin_edges = np.histogram(image.ravel(), bins=256, range=(0.0, 1.0))
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    total = hist.sum()
    if total == 0:
        return 0.5

    weight1 = np.cumsum(hist)
    weight2 = total - weight1
    mean1 = np.cumsum(hist * bin_centers) / np.maximum(weight1, 1)
    mean2 = (np.cumsum((hist * bin_centers)[::-1])[::-1]) / np.maximum(weight2, 1)

    var_between = weight1 * weight2 * (mean1 - mean2) ** 2
    idx = np.argmax(var_between)
    return float(bin_centers[idx])


def _morphological_erosion(mask: np.ndarray, radius: int = 3) -> np.ndarray:
    """Simple square erosion using NumPy strides."""
    if radius == 0:
        return mask.copy()
    h, w = mask.shape
    result = np.zeros_like(mask)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            shifted = np.roll(np.roll(mask, dy, axis=0), dx, axis=1)
            result = result | shifted
    return result & mask


def _morphological_dilation(mask: np.ndarray, radius: int = 3) -> np.ndarray:
    """Simple square dilation."""
    if radius == 0:
        return mask.copy()
    result = np.zeros_like(mask, dtype=bool)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            shifted = np.roll(np.roll(mask, dy, axis=0), dx, axis=1)
            result = result | shifted
    return result


def _connected_components(binary: np.ndarray) -> np.ndarray:
    """Simple flood-fill connected components labeling (8-connectivity).

    Returns label array where 0=background, >0 = component label.
    """
    labels = np.zeros_like(binary, dtype=np.int32)
    label_idx = 0
    rows, cols = np.where(binary)

    visited = np.zeros_like(binary, dtype=bool)
    h, w = binary.shape

    for r, c in zip(rows.tolist(), cols.tolist()):
        if visited[r, c]:
            continue
        label_idx += 1
        stack = [(r, c)]
        while stack:
            cr, cc = stack.pop()
            if cr < 0 or cr >= h or cc < 0 or cc >= w:
                continue
            if visited[cr, cc] or not binary[cr, cc]:
                continue
            visited[cr, cc] = True
            labels[cr, cc] = label_idx
            stack.extend([
                (cr-1, cc), (cr+1, cc), (cr, cc-1), (cr, cc+1),
                (cr-1, cc-1), (cr-1, cc+1), (cr+1, cc-1), (cr+1, cc+1),
            ])
    return labels


def _largest_component_mask(binary: np.ndarray) -> np.ndarray:
    """Return binary mask containing only the largest connected component."""
    labels = _connected_components(binary)
    if labels.max() == 0:
        return np.zeros_like(binary, dtype=bool)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0  # ignore background
    largest = int(sizes.argmax())
    return labels == largest


# ---------------------------------------------------------------------------
# Optic Disc Segmentation
# ---------------------------------------------------------------------------

class OpticDiscSegmenter:
    """
    Segment the optic disc from a fundus image using the bright region approach.

    The optic disc is the brightest circular region in the fundus, typically
    located in the nasal part of the retina.  This implementation:
      1. Extracts the green channel (best contrast for OD)
      2. Applies Gaussian smoothing via NumPy convolution
      3. Thresholds to isolate bright regions
      4. Selects the largest connected component
      5. Computes a tight bounding box
    """

    def __init__(self, percentile: float = 90.0):
        self.percentile = percentile  # Use top N% pixels as OD candidates

    def segment(self, image: np.ndarray) -> Tuple[np.ndarray, Optional[BoundingBox]]:
        """
        Segment optic disc from RGB fundus image.

        Args:
            image: uint8 RGB array of shape (H, W, 3)

        Returns:
            (binary_mask, bounding_box) — mask is bool array, bbox may be None
        """
        if image.ndim != 3 or image.shape[2] < 3:
            raise ValueError("Expected RGB image with shape (H, W, 3)")

        # Use the green channel — best disc contrast in fundus
        green = image[:, :, 1].astype(np.float64) / 255.0
        smooth = self._smooth(green, kernel_size=15)
        normalized = _normalize(smooth)

        threshold = np.percentile(normalized, self.percentile)
        binary = normalized >= threshold

        # Keep only the largest bright region (optic disc)
        if binary.any():
            disc_mask = _largest_component_mask(binary)
        else:
            disc_mask = np.zeros_like(binary)

        bbox = self._compute_bbox(disc_mask) if disc_mask.any() else None
        return disc_mask, bbox

    @staticmethod
    def _smooth(image: np.ndarray, kernel_size: int = 15) -> np.ndarray:
        """Vectorized Box-filter smoothing."""
        k = kernel_size
        pad = k // 2
        padded = np.pad(image, pad, mode="reflect")
        
        # Use sliding_window_view for vectorized convolution
        # (modern NumPy 1.20+)
        from numpy.lib.stride_tricks import sliding_window_view
        windows = sliding_window_view(padded, (k, k))
        return windows.mean(axis=(2, 3))

    @staticmethod
    def _compute_bbox(mask: np.ndarray) -> Optional[BoundingBox]:
        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        if len(rows) == 0 or len(cols) == 0:
            return None
        return BoundingBox(
            x=int(cols[0]), y=int(rows[0]),
            w=int(cols[-1] - cols[0] + 1),
            h=int(rows[-1] - rows[0] + 1),
        )


# ---------------------------------------------------------------------------
# Blood Vessel Segmentation
# ---------------------------------------------------------------------------

class BloodVesselSegmenter:
    """
    Segment retinal blood vessels using the green channel with local contrast
    enhancement and adaptive thresholding.

    Algorithm:
      1. Extract green channel (best vessel contrast)
      2. Apply CLAHE-like local normalization
      3. Invert (vessels are dark on bright background)
      4. Threshold with Otsu's method
      5. Morphological cleanup
    """

    def __init__(self, block_size: int = 64, threshold_offset: float = 0.05):
        self.block_size = block_size
        self.threshold_offset = threshold_offset

    def segment(self, image: np.ndarray) -> np.ndarray:
        """
        Segment blood vessels from RGB fundus image.

        Args:
            image: uint8 RGB array of shape (H, W, 3)

        Returns:
            Binary bool mask — True where vessel detected
        """
        if image.ndim != 3 or image.shape[2] < 3:
            raise ValueError("Expected RGB image with shape (H, W, 3)")

        green = image[:, :, 1].astype(np.float64) / 255.0
        enhanced = self._local_contrast_enhance(green)
        inverted = 1.0 - enhanced  # Vessels are dark → invert

        threshold = _threshold_otsu(inverted) - self.threshold_offset
        binary = inverted > max(0.0, threshold)

        # Morphological cleanup
        cleaned = _morphological_erosion(binary.astype(np.uint8), radius=1).astype(bool)
        return cleaned

    def _local_contrast_enhance(self, image: np.ndarray) -> np.ndarray:
        """Block-wise local contrast normalization."""
        h, w = image.shape
        result = np.zeros_like(image)
        bs = self.block_size

        for i in range(0, h, bs):
            for j in range(0, w, bs):
                block = image[i:i+bs, j:j+bs]
                mn, mx = block.min(), block.max()
                if mx - mn < 1e-6:
                    result[i:i+bs, j:j+bs] = block
                else:
                    result[i:i+bs, j:j+bs] = (block - mn) / (mx - mn)
        return result


# ---------------------------------------------------------------------------
# Lesion Segmentation (bright lesions: exudates, drusen)
# ---------------------------------------------------------------------------

class LesionSegmenter:
    """
    Detect bright lesions (exudates, cotton wool spots, drusen) in fundus images.

    Bright lesions appear whiter than the surrounding retinal background.
    The optic disc (the brightest region) is excluded using a disc mask.

    Algorithm:
      1. Convert to grayscale
      2. Subtract smoothed background (local mean)
      3. Threshold residual bright regions
      4. Exclude optic disc region
      5. Remove small noise components
    """

    def __init__(
        self,
        brightness_percentile: float = 92.0,
        min_lesion_size: int = 20,
    ):
        self.brightness_percentile = brightness_percentile
        self.min_lesion_size = min_lesion_size

    def segment(
        self,
        image: np.ndarray,
        disc_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Detect lesions in fundus image.

        Args:
            image: uint8 RGB array of shape (H, W, 3)
            disc_mask: Optional bool mask of optic disc region to exclude

        Returns:
            (lesion_mask, lesion_count) — mask True where lesion detected
        """
        gray = _rgb_to_gray(image)
        background = self._estimate_background(gray)
        residual = _normalize(gray - background)

        threshold = np.percentile(residual, self.brightness_percentile)
        binary = residual > threshold

        # Exclude optic disc
        if disc_mask is not None:
            dilated_disc = _morphological_dilation(disc_mask.astype(bool), radius=5)
            binary = binary & ~dilated_disc

        # Remove small components
        labels = _connected_components(binary)
        lesion_mask = np.zeros_like(binary, dtype=bool)
        count = 0

        if labels.max() > 0:
            sizes = np.bincount(labels.ravel())
            sizes[0] = 0
            for label_id in range(1, labels.max() + 1):
                if sizes[label_id] >= self.min_lesion_size:
                    lesion_mask |= (labels == label_id)
                    count += 1

        return lesion_mask, count

    @staticmethod
    def _estimate_background(image: np.ndarray, window: int = 31) -> np.ndarray:
        """Vectorized background estimation via box filter."""
        from numpy.lib.stride_tricks import sliding_window_view
        pad = window // 2
        padded = np.pad(image, pad, mode="reflect")
        windows = sliding_window_view(padded, (window, window))
        return windows.mean(axis=(2, 3))


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

class FundusImagingSegmenter:
    """
    Full pipeline: optic disc + blood vessels + lesion segmentation.

    Usage::

        segmenter = FundusImagingSegmenter()
        image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        result = segmenter.segment(image, target=SegmentationTarget.ALL)
        print(result.to_dict())
    """

    def __init__(
        self,
        disc_percentile: float = 90.0,
        vessel_block_size: int = 64,
        lesion_percentile: float = 92.0,
        min_lesion_size: int = 20,
    ):
        self.disc_segmenter = OpticDiscSegmenter(percentile=disc_percentile)
        self.vessel_segmenter = BloodVesselSegmenter(block_size=vessel_block_size)
        self.lesion_segmenter = LesionSegmenter(
            brightness_percentile=lesion_percentile,
            min_lesion_size=min_lesion_size,
        )

    def segment(
        self,
        image: np.ndarray,
        target: SegmentationTarget = SegmentationTarget.ALL,
    ) -> SegmentationResult:
        """
        Run segmentation pipeline on a fundus image.

        Args:
            image: uint8 RGB numpy array (H, W, 3)
            target: which structures to segment

        Returns:
            SegmentationResult with populated masks and metrics
        """
        if image.ndim != 3 or image.shape[2] < 3:
            raise ValueError(f"Expected RGB image (H, W, 3), got shape {image.shape}")
        if image.dtype != np.uint8:
            raise TypeError(f"Expected uint8 image, got {image.dtype}")

        result = SegmentationResult(image_shape=tuple(image.shape))  # type: ignore[arg-type]

        # Optic disc
        if target in {SegmentationTarget.OPTIC_DISC, SegmentationTarget.ALL}:
            logger.info("Segmenting optic disc...")
            disc_mask, disc_bbox = self.disc_segmenter.segment(image)
            result.optic_disc_mask = disc_mask
            result.optic_disc_bbox = disc_bbox
            if disc_mask.any():
                result.metrics["disc_area_pixels"] = float(disc_mask.sum())
                total = image.shape[0] * image.shape[1]
                result.metrics["disc_area_fraction"] = float(disc_mask.sum()) / total

        # Blood vessels
        if target in {SegmentationTarget.BLOOD_VESSELS, SegmentationTarget.ALL}:
            logger.info("Segmenting blood vessels...")
            vessel_mask = self.vessel_segmenter.segment(image)
            result.vessel_mask = vessel_mask
            if vessel_mask.any():
                total = image.shape[0] * image.shape[1]
                result.metrics["vessel_density"] = float(vessel_mask.sum()) / total

        # Lesions
        if target in {SegmentationTarget.LESIONS, SegmentationTarget.ALL}:
            logger.info("Detecting lesions...")
            disc_mask_ref = result.optic_disc_mask
            lesion_mask, count = self.lesion_segmenter.segment(image, disc_mask=disc_mask_ref)
            result.lesion_mask = lesion_mask
            result.lesion_count = count
            result.metrics["lesion_count"] = float(count)
            if lesion_mask.any():
                total = image.shape[0] * image.shape[1]
                result.metrics["lesion_area_fraction"] = float(lesion_mask.sum()) / total

        logger.info(f"Segmentation complete. Metrics: {result.metrics}")
        return result


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def dice_coefficient(pred: np.ndarray, ground_truth: np.ndarray) -> float:
    """Compute Sørensen–Dice coefficient between two binary masks.

    Args:
        pred: Predicted binary mask (bool or 0/1 int)
        ground_truth: Reference binary mask

    Returns:
        Dice score in [0, 1] — 1.0 = perfect overlap
    """
    pred = pred.astype(bool)
    gt = ground_truth.astype(bool)
    intersection = (pred & gt).sum()
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0  # Both empty → perfect agreement
    return float(2.0 * intersection / denom)


def jaccard_index(pred: np.ndarray, ground_truth: np.ndarray) -> float:
    """Compute Jaccard (IoU) index between two binary masks.

    Args:
        pred: Predicted binary mask
        ground_truth: Reference binary mask

    Returns:
        IoU score in [0, 1]
    """
    pred = pred.astype(bool)
    gt = ground_truth.astype(bool)
    intersection = (pred & gt).sum()
    union = (pred | gt).sum()
    if union == 0:
        return 1.0
    return float(intersection / union)


def sensitivity(pred: np.ndarray, ground_truth: np.ndarray) -> float:
    """Recall / sensitivity = TP / (TP + FN)."""
    pred = pred.astype(bool)
    gt = ground_truth.astype(bool)
    tp = (pred & gt).sum()
    fn = (~pred & gt).sum()
    if tp + fn == 0:
        return 1.0
    return float(tp / (tp + fn))


def specificity(pred: np.ndarray, ground_truth: np.ndarray) -> float:
    """Specificity = TN / (TN + FP)."""
    pred = pred.astype(bool)
    gt = ground_truth.astype(bool)
    tn = (~pred & ~gt).sum()
    fp = (pred & ~gt).sum()
    if tn + fp == 0:
        return 1.0
    return float(tn / (tn + fp))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def print_clinical_summary(result: SegmentationResult) -> None:
    print("\n" + "═" * 50)
    print("  FUNDUS SCREENING CLINICAL SUMMARY")
    print("═" * 50)
    
    metrics = result.metrics
    
    print(f"  IMAGE DATA")
    print(f"  ├─ Resolution : {result.width} x {result.height}")
    print(f"  └─ Status     : {'PROCESSED'}")
    
    print(f"\n  ANATOMY")
    od_status = "FOUND" if result.optic_disc_bbox else "NOT DETECTED"
    print(f"  ├─ Optic Disc : {od_status}")
    if result.optic_disc_bbox:
        print(f"  │  └─ Area    : {metrics.get('disc_area_fraction', 0)*100:>.2f}% of retina")
    print(f"  └─ Vasculature: {metrics.get('vessel_density', 0)*100:>.2f}% density")
    
    print(f"\n  PATHOLOGY")
    print(f"  ├─ Lesions    : {int(result.lesion_count)}")
    level = "NORMAL" if result.lesion_count == 0 else "ACTION REQUIRED" if result.lesion_count > 5 else "MONITOR"
    print(f"  └─ Risk Level : {level}")
    
    print("═" * 50)
    print(f"  Disclaimer: AI-generated screening aid. Not for diagnosis.")
    print("═" * 50 + "\n")


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="fundus-imaging-segmenter",
        description="Retinal fundus image segmentation tool",
    )
    subparsers = parser.add_subparsers(dest="command")

    # segment command
    seg = subparsers.add_parser("segment", help="Segment a fundus image (numpy .npy file)")
    seg.add_argument("image", help="Path to uint8 RGB numpy image (.npy)")
    seg.add_argument(
        "--target", choices=["optic_disc", "blood_vessels", "lesions", "all"],
        default="all",
    )
    seg.add_argument("--json", action="store_true", dest="json_output")
    seg.add_argument("--summary", action="store_true", help="Print clinical summary")

    # demo command
    demo = subparsers.add_parser("demo", help="Run demo on a synthetic image")
    demo.add_argument("--summary", action="store_true", help="Print clinical summary")

    # metrics command
    met = subparsers.add_parser("metrics", help="Evaluate segmentation against ground truth")
    met.add_argument("pred", help="Predicted mask (.npy, bool or int)")
    met.add_argument("gt", help="Ground truth mask (.npy, bool or int)")

    args = parser.parse_args(argv)

    if args.command == "segment":
        try:
            image = np.load(args.image)
        except Exception as e:
            print(f"❌ Cannot load image: {e}", file=sys.stderr)
            return 1
        if image.dtype != np.uint8:
            image = (image * 255).clip(0, 255).astype(np.uint8)

        target = SegmentationTarget(args.target)
        segmenter = FundusImagingSegmenter()
        result = segmenter.segment(image, target=target)

        if args.json_output:
            print(json.dumps(result.to_dict(), indent=2))
        elif args.summary:
            print_clinical_summary(result)
        else:
            print("\n=== Fundus Segmentation Result ===")
            r = result.to_dict()
            print(f"  Image shape  : {r['image_shape']}")
            print(f"  Optic disc   : bbox={r['optic_disc_bbox']}")
            print(f"  Vessel mask  : {r['has_vessel_mask']}")
            print(f"  Lesion count : {r['lesion_count']}")
            print(f"  Metrics      : {r['metrics']}")
        return 0

    elif args.command == "demo":
        print("Running demo on synthetic 256×256 fundus image...")
        rng = np.random.default_rng(42)
        image = rng.integers(60, 180, (256, 256, 3), dtype=np.uint8)
        # Simulate bright optic disc region
        image[80:140, 100:160] = [230, 220, 190]
        # Simulate some vessels (dark lines)
        image[50:200, 128] = [40, 70, 40]
        image[128, 50:200] = [40, 70, 40]

        segmenter = FundusImagingSegmenter()
        result = segmenter.segment(image, target=SegmentationTarget.ALL)

        if args.summary:
            print_clinical_summary(result)
        else:
            print(f"  Shape        : {result.image_shape}")
            print(f"  Disc bbox    : {result.optic_disc_bbox}")
            print(f"  Vessel pixels: {result.vessel_mask.sum() if result.vessel_mask is not None else 0}")
            print(f"  Lesion count : {result.lesion_count}")
            print(f"  Metrics      : {result.metrics}")
        return 0

    elif args.command == "metrics":
        try:
            pred = np.load(args.pred).astype(bool)
            gt = np.load(args.gt).astype(bool)
        except Exception as e:
            print(f"❌ Cannot load masks: {e}", file=sys.stderr)
            return 1

        scores = {
            "dice": dice_coefficient(pred, gt),
            "jaccard": jaccard_index(pred, gt),
            "sensitivity": sensitivity(pred, gt),
            "specificity": specificity(pred, gt),
        }
        print(json.dumps(scores, indent=2))
        return 0

    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
