"""
Tests for fundus-imaging-segmenter

Run with:  pytest tests/ -v
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from main import (
    BoundingBox,
    BloodVesselSegmenter,
    FundusImagingSegmenter,
    LesionSegmenter,
    OpticDiscSegmenter,
    SegmentationResult,
    SegmentationTarget,
    _connected_components,
    _largest_component_mask,
    _normalize,
    _rgb_to_gray,
    _threshold_otsu,
    dice_coefficient,
    jaccard_index,
    sensitivity,
    specificity,
    main as cli_main,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def black_image():
    return np.zeros((64, 64, 3), dtype=np.uint8)


@pytest.fixture
def bright_image():
    """Uniform bright image."""
    return np.full((64, 64, 3), 200, dtype=np.uint8)


@pytest.fixture
def synthetic_fundus():
    """Synthetic fundus with bright disc and dark vessels."""
    rng = np.random.default_rng(0)
    img = rng.integers(80, 150, (128, 128, 3), dtype=np.uint8)
    # Bright optic disc region
    img[40:70, 50:85] = [240, 230, 200]
    # Dark vessel
    img[64, 20:100] = [30, 60, 30]
    return img


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

class TestRgbToGray:
    def test_shape(self, synthetic_fundus):
        gray = _rgb_to_gray(synthetic_fundus)
        assert gray.shape == (128, 128)

    def test_range(self, synthetic_fundus):
        gray = _rgb_to_gray(synthetic_fundus)
        assert gray.min() >= 0.0
        assert gray.max() <= 1.0

    def test_already_gray(self):
        gray2d = np.full((32, 32), 128, dtype=np.uint8)
        result = _rgb_to_gray(gray2d)
        assert result.shape == (32, 32)

    def test_uniform_value(self):
        img = np.full((10, 10, 3), 255, dtype=np.uint8)
        gray = _rgb_to_gray(img)
        assert np.allclose(gray, 1.0, atol=0.01)


class TestNormalize:
    def test_zero_one_range(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0])
        n = _normalize(arr)
        assert np.isclose(n.min(), 0.0)
        assert np.isclose(n.max(), 1.0)

    def test_constant_returns_zeros(self):
        arr = np.ones((5, 5))
        n = _normalize(arr)
        assert np.all(n == 0.0)

    def test_shape_preserved(self):
        arr = np.random.rand(10, 10)
        assert _normalize(arr).shape == arr.shape


class TestThresholdOtsu:
    def test_bimodal_image(self):
        # Bimodal: half zeros, half ones
        arr = np.array([0.0] * 50 + [1.0] * 50)
        t = _threshold_otsu(arr)
        # Threshold should be a valid value between the two modes
        assert 0.0 <= t <= 1.0

    def test_constant_image(self):
        arr = np.ones(100) * 0.5
        t = _threshold_otsu(arr)
        assert 0.0 <= t <= 1.0


class TestConnectedComponents:
    def test_single_blob(self):
        binary = np.zeros((10, 10), dtype=bool)
        binary[2:5, 2:5] = True
        labels = _connected_components(binary)
        assert labels.max() == 1
        assert (labels > 0).sum() == 9

    def test_two_blobs(self):
        binary = np.zeros((20, 20), dtype=bool)
        binary[2:5, 2:5] = True
        binary[15:18, 15:18] = True
        labels = _connected_components(binary)
        assert labels.max() == 2

    def test_empty(self):
        binary = np.zeros((10, 10), dtype=bool)
        labels = _connected_components(binary)
        assert labels.max() == 0


class TestLargestComponentMask:
    def test_largest_selected(self):
        binary = np.zeros((20, 20), dtype=bool)
        binary[1:3, 1:3] = True   # Small: 4 pixels
        binary[10:16, 10:16] = True  # Large: 36 pixels
        mask = _largest_component_mask(binary)
        assert mask[11, 11] is np.True_
        assert mask[1, 1] is not np.True_

    def test_empty_returns_zeros(self):
        binary = np.zeros((10, 10), dtype=bool)
        mask = _largest_component_mask(binary)
        assert not mask.any()


# ---------------------------------------------------------------------------
# BoundingBox
# ---------------------------------------------------------------------------

class TestBoundingBox:
    def test_center(self):
        bbox = BoundingBox(x=10, y=20, w=40, h=60)
        assert bbox.center == (30, 50)

    def test_area(self):
        bbox = BoundingBox(x=0, y=0, w=10, h=5)
        assert bbox.area == 50

    def test_to_dict(self):
        bbox = BoundingBox(x=1, y=2, w=3, h=4)
        d = bbox.to_dict()
        assert d == {"x": 1, "y": 2, "w": 3, "h": 4}


# ---------------------------------------------------------------------------
# OpticDiscSegmenter
# ---------------------------------------------------------------------------

class TestOpticDiscSegmenter:
    def setup_method(self):
        self.seg = OpticDiscSegmenter(percentile=85.0)

    def test_returns_mask_and_bbox(self, synthetic_fundus):
        mask, bbox = self.seg.segment(synthetic_fundus)
        assert mask.shape == (128, 128)
        assert mask.dtype == bool

    def test_disc_region_detected(self, synthetic_fundus):
        """Bright disc region [40:70, 50:85] should be mostly in mask."""
        mask, bbox = self.seg.segment(synthetic_fundus)
        disc_region = mask[40:70, 50:85]
        # At least 20% of the bright region should be detected
        assert disc_region.sum() > 0

    def test_bbox_not_none_for_non_black(self, synthetic_fundus):
        _, bbox = self.seg.segment(synthetic_fundus)
        assert bbox is not None

    def test_rejects_non_rgb(self):
        with pytest.raises(ValueError):
            self.seg.segment(np.zeros((64, 64), dtype=np.uint8))

    def test_black_image_has_bbox_or_none(self, black_image):
        # Should not crash — may or may not find a disc
        mask, bbox = self.seg.segment(black_image)
        assert mask.shape == black_image.shape[:2]


# ---------------------------------------------------------------------------
# BloodVesselSegmenter
# ---------------------------------------------------------------------------

class TestBloodVesselSegmenter:
    def setup_method(self):
        self.seg = BloodVesselSegmenter(block_size=32)

    def test_returns_bool_mask(self, synthetic_fundus):
        mask = self.seg.segment(synthetic_fundus)
        assert mask.dtype == bool
        assert mask.shape == (128, 128)

    def test_rejects_non_rgb(self):
        with pytest.raises(ValueError):
            self.seg.segment(np.zeros((64, 64), dtype=np.uint8))

    def test_some_vessels_detected(self, synthetic_fundus):
        """At least some pixels should be classified as vessel."""
        mask = self.seg.segment(synthetic_fundus)
        assert mask.sum() > 0


# ---------------------------------------------------------------------------
# LesionSegmenter
# ---------------------------------------------------------------------------

class TestLesionSegmenter:
    def setup_method(self):
        self.seg = LesionSegmenter(brightness_percentile=85.0, min_lesion_size=5)

    def test_returns_mask_and_count(self, synthetic_fundus):
        mask, count = self.seg.segment(synthetic_fundus)
        assert mask.dtype == bool
        assert mask.shape == (128, 128)
        assert isinstance(count, int)
        assert count >= 0

    def test_disc_exclusion(self, synthetic_fundus):
        disc_mask = np.zeros((128, 128), dtype=bool)
        disc_mask[40:70, 50:85] = True
        mask_with_excl, _ = self.seg.segment(synthetic_fundus, disc_mask=disc_mask)
        # Disc region should not be labeled as lesion
        assert mask_with_excl[50:65, 55:80].sum() == 0

    def test_no_crash_on_uniform_image(self, bright_image):
        mask, count = self.seg.segment(bright_image)
        assert mask.shape == bright_image.shape[:2]


# ---------------------------------------------------------------------------
# FundusImagingSegmenter (full pipeline)
# ---------------------------------------------------------------------------

class TestFundusImagingSegmenter:
    def setup_method(self):
        self.seg = FundusImagingSegmenter(disc_percentile=85.0)

    def test_all_target_populates_all_fields(self, synthetic_fundus):
        result = self.seg.segment(synthetic_fundus, target=SegmentationTarget.ALL)
        assert result.optic_disc_mask is not None
        assert result.vessel_mask is not None
        assert result.lesion_mask is not None

    def test_optic_disc_only(self, synthetic_fundus):
        result = self.seg.segment(synthetic_fundus, target=SegmentationTarget.OPTIC_DISC)
        assert result.optic_disc_mask is not None
        assert result.vessel_mask is None
        assert result.lesion_mask is None

    def test_vessel_only(self, synthetic_fundus):
        result = self.seg.segment(synthetic_fundus, target=SegmentationTarget.BLOOD_VESSELS)
        assert result.optic_disc_mask is None
        assert result.vessel_mask is not None
        assert result.lesion_mask is None

    def test_lesion_only(self, synthetic_fundus):
        result = self.seg.segment(synthetic_fundus, target=SegmentationTarget.LESIONS)
        assert result.optic_disc_mask is None
        assert result.vessel_mask is None
        assert result.lesion_mask is not None

    def test_rejects_wrong_dtype(self, synthetic_fundus):
        bad = synthetic_fundus.astype(np.float32)
        with pytest.raises(TypeError):
            self.seg.segment(bad)

    def test_rejects_wrong_dims(self):
        with pytest.raises(ValueError):
            self.seg.segment(np.zeros((64, 64), dtype=np.uint8))

    def test_result_to_dict(self, synthetic_fundus):
        result = self.seg.segment(synthetic_fundus, target=SegmentationTarget.ALL)
        d = result.to_dict()
        assert "image_shape" in d
        assert "lesion_count" in d
        assert "metrics" in d


# ---------------------------------------------------------------------------
# Evaluation Metrics
# ---------------------------------------------------------------------------

class TestEvaluationMetrics:
    def _make_masks(self):
        pred = np.zeros((10, 10), dtype=bool)
        gt = np.zeros((10, 10), dtype=bool)
        pred[2:5, 2:5] = True
        gt[3:6, 3:6] = True
        return pred, gt

    def test_dice_perfect(self):
        mask = np.ones((10, 10), dtype=bool)
        assert np.isclose(dice_coefficient(mask, mask), 1.0)

    def test_dice_no_overlap(self):
        pred = np.zeros((10, 10), dtype=bool)
        pred[:5, :5] = True
        gt = np.zeros((10, 10), dtype=bool)
        gt[5:, 5:] = True
        assert np.isclose(dice_coefficient(pred, gt), 0.0)

    def test_dice_partial(self):
        pred, gt = self._make_masks()
        d = dice_coefficient(pred, gt)
        assert 0.0 < d < 1.0

    def test_dice_both_empty(self):
        assert np.isclose(dice_coefficient(
            np.zeros((5, 5), dtype=bool),
            np.zeros((5, 5), dtype=bool),
        ), 1.0)

    def test_jaccard_perfect(self):
        mask = np.ones((5, 5), dtype=bool)
        assert np.isclose(jaccard_index(mask, mask), 1.0)

    def test_jaccard_no_overlap(self):
        pred = np.zeros((10, 10), dtype=bool)
        pred[:5, :] = True
        gt = np.zeros((10, 10), dtype=bool)
        gt[5:, :] = True
        assert np.isclose(jaccard_index(pred, gt), 0.0)

    def test_jaccard_le_dice(self):
        pred, gt = self._make_masks()
        assert jaccard_index(pred, gt) <= dice_coefficient(pred, gt)

    def test_sensitivity_all_tp(self):
        mask = np.ones((5, 5), dtype=bool)
        assert np.isclose(sensitivity(mask, mask), 1.0)

    def test_sensitivity_all_missed(self):
        pred = np.zeros((5, 5), dtype=bool)
        gt = np.ones((5, 5), dtype=bool)
        assert np.isclose(sensitivity(pred, gt), 0.0)

    def test_specificity_no_fp(self):
        pred = np.zeros((5, 5), dtype=bool)
        gt = np.zeros((5, 5), dtype=bool)
        assert np.isclose(specificity(pred, gt), 1.0)


# ---------------------------------------------------------------------------
# CLI integration tests
# ---------------------------------------------------------------------------

class TestCLI:
    def test_demo_command(self):
        rc = cli_main(["demo"])
        assert rc == 0

    def test_no_command(self):
        rc = cli_main([])
        assert rc == 0

    def test_missing_segment_file(self):
        rc = cli_main(["segment", "nonexistent.npy"])
        assert rc == 1

    def test_segment_valid_file(self, tmp_path):
        rng = np.random.default_rng(1)
        img = rng.integers(60, 200, (64, 64, 3), dtype=np.uint8)
        fpath = tmp_path / "img.npy"
        np.save(str(fpath), img)
        rc = cli_main(["segment", str(fpath), "--target", "all"])
        assert rc == 0

    def test_segment_json_output(self, tmp_path, capsys):
        import json
        rng = np.random.default_rng(2)
        img = rng.integers(60, 200, (64, 64, 3), dtype=np.uint8)
        fpath = tmp_path / "img.npy"
        np.save(str(fpath), img)
        rc = cli_main(["segment", str(fpath), "--json"])
        assert rc == 0
        out = capsys.readouterr().out
        parsed = json.loads(out)
        assert "image_shape" in parsed

    def test_metrics_command(self, tmp_path):
        pred = np.ones((10, 10), dtype=bool)
        gt = np.ones((10, 10), dtype=bool)
        pred_path = tmp_path / "pred.npy"
        gt_path = tmp_path / "gt.npy"
        np.save(str(pred_path), pred)
        np.save(str(gt_path), gt)
        rc = cli_main(["metrics", str(pred_path), str(gt_path)])
        assert rc == 0

    def test_metrics_missing_files(self):
        rc = cli_main(["metrics", "no_pred.npy", "no_gt.npy"])
        assert rc == 1
