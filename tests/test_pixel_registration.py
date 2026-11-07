"""Regression checks for false strip matches and subpixel scan registration.

Run with PYTHONPATH=src python -m unittest discover -s tests -v.
"""

import unittest
from pathlib import Path

import cv2 as cv
import numpy as np

from stitching.feature_matching import PairwiseMatch
# noinspection protected-member
from stitching.pixel_registration import (
    _make_samples,
    _maximum_pose_movement,
    _PixelObjective,
    prepare_pixel_images,
    refine_global_pixels,
    validate_pixel_pairs,
)
from stitching.shared import StitchingConfig, StitchingImages
from stitching.transformation_estimation import estimate_global_transforms


def _pattern(width: int = 640, height: int = 480) -> np.ndarray:
    rng = np.random.default_rng(42)
    image = np.full((height, width, 3), 245, dtype=np.uint8)
    for _ in range(140):
        start = rng.integers((15, 15), (width - 15, height - 15))
        end = np.clip(start + rng.integers(-45, 46, 2), 5, (width - 5, height - 5))
        cv.line(image, tuple(start), tuple(end), (25, 25, 25), int(rng.integers(1, 4)), cv.LINE_AA)
    for number in range(15):
        origin = tuple(rng.integers((20, 20), (width - 80, height - 20)))
        cv.putText(image, str(number), origin, cv.FONT_HERSHEY_SIMPLEX, .6, (30, 30, 30), 1, cv.LINE_AA)
    return image


def _images(*images: np.ndarray) -> StitchingImages:
    return StitchingImages(tuple(Path(f"scan_{index}.png") for index in range(len(images))), images, (None,) * len(images))


def _pair(i: int, j: int, points: np.ndarray, targets: np.ndarray) -> PairwiseMatch:
    return PairwiseMatch(i, j, len(points), len(points), 2., None, points, targets, None)


def _corner_error(actual: np.ndarray, expected: np.ndarray, image: np.ndarray) -> float:
    height, width = image.shape[:2]
    corners = np.array(((0, 0), (width - 1, 0), (0, height - 1), (width - 1, height - 1)))
    errors = corners @ (actual[:2, :2] - expected[:2, :2]).T + actual[:2, 2] - expected[:2, 2]
    return float(np.max(np.linalg.norm(errors, axis=1)))


class PixelRegistrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cv.setNumThreads(1)
        cls.image = _pattern()
        cls.points = np.random.default_rng(8).uniform((60, 60), (580, 420), (80, 2))
        cls.config = StitchingConfig(irls_iterations=1, pixel_max_samples=6000)

    def test_corrects_biased_features_to_subpixel_translation_and_rotation(self) -> None:
        angle = np.deg2rad(.25)
        cosine, sine = np.cos(angle), np.sin(angle)
        actual = np.array(((cosine, -sine, 3.7), (sine, cosine, -2.4), (0., 0., 1.)))
        moving = cv.warpAffine(self.image, actual[:2], (640, 480), borderValue=(245, 245, 245))
        images = _images(self.image, moving)
        targets = self.points @ actual[:2, :2].T + actual[:2, 2] + (2.8, -1.7)
        pair = _pair(0, 1, self.points, targets)
        feature = estimate_global_transforms((pair,), images, StitchingConfig(pixel_refinement=False))
        refined = estimate_global_transforms((pair,), images, self.config)
        expected = np.linalg.inv(actual)
        self.assertGreater(_corner_error(feature[1], expected, moving), 3.)
        self.assertLess(_corner_error(refined[1], expected, moving), .15)
        np.testing.assert_array_equal(refined[0], np.eye(3))
        np.testing.assert_allclose(refined[1][:2, :2].T @ refined[1][:2, :2], np.eye(2), atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(refined[1][:2, :2]), 1., places=12)

    def test_handles_quarter_turn_and_brightness_difference(self) -> None:
        actual = np.array(((0., -1., 479.), (1., 0., 0.), (0., 0., 1.)))
        moving = cv.convertScaleAbs(cv.rotate(self.image, cv.ROTATE_90_CLOCKWISE), alpha=.85, beta=17)
        targets = self.points @ actual[:2, :2].T + actual[:2, 2] + (2.1, -1.8)
        transforms = estimate_global_transforms((_pair(0, 1, self.points, targets),), _images(self.image, moving), self.config)
        self.assertLess(_corner_error(transforms[1], np.linalg.inv(actual), moving), .15)

    def test_alignment_does_not_require_a_white_background(self) -> None:
        pattern = (245. - self.image[:, :, 0].astype(np.float32)) / 220.
        angle = np.deg2rad(.25)
        actual = np.array(((np.cos(angle), -np.sin(angle), 3.7), (np.sin(angle), np.cos(angle), -2.4), (0., 0., 1.)))
        targets = self.points @ actual[:2, :2].T + actual[:2, 2] + (2.8, -1.7)
        pair = _pair(0, 1, self.points, targets)
        for background, foreground in (
                ((180, 125, 70), (20, 20, 20)),
                ((130, 130, 130), (15, 15, 15)),
                ((15, 15, 15), (235, 235, 235)),
                ((70, 30, 15), (230, 230, 230)),
        ):
            with self.subTest(background=background, foreground=foreground):
                base, ink = np.asarray(background, np.float32), np.asarray(foreground, np.float32)
                image = np.clip(np.rint(base + pattern[:, :, None] * (ink - base)), 0, 255).astype(np.uint8)
                moving = cv.warpAffine(image, actual[:2], (640, 480), borderValue=background)
                transforms = estimate_global_transforms((pair,), _images(image, moving), self.config)
                self.assertLess(_corner_error(transforms[1], np.linalg.inv(actual), moving), .15)

    def test_removes_coherent_false_edge_and_refines_the_connected_graph(self) -> None:
        canvas = _pattern(960)
        images = _images(canvas[:, :480].copy(), canvas[:, 240:720].copy(), canvas[:, 480:].copy())
        points = np.random.default_rng(17).uniform((270, 60), (440, 420), (80, 2))
        matches = (
            _pair(0, 1, points, points - (240., 0.) + (2.1, -1.4)),
            _pair(0, 2, points, points.copy()),
            _pair(1, 2, points, points - (240., 0.) + (1.7, 1.2)),
        )
        verified = validate_pixel_pairs(matches, prepare_pixel_images(images), self.config)
        self.assertEqual([pair.accepted for pair in verified], [True, False, True])
        transforms = estimate_global_transforms(matches, images, self.config)
        for index, transform in enumerate(transforms):
            expected = np.eye(3)
            expected[0, 2] = 240. * index
            self.assertLess(_corner_error(transform, expected, images.scan_images[index]), .15)

    def test_uniform_backgrounds_are_not_matching_evidence_and_disconnect_is_explicit(self) -> None:
        pair = _pair(0, 1, self.points, self.points.copy())
        for background in ((245, 245, 245), (180, 125, 70), (130, 130, 130), (15, 15, 15)):
            with self.subTest(background=background):
                blank = np.full_like(self.image, background)
                with self.assertRaisesRegex(RuntimeError, r"Pixel validation.*Disconnected match graph.*\[1\]"):
                    estimate_global_transforms((pair,), _images(blank, blank.copy()), self.config)

    def test_feature_transform_with_no_geometric_overlap_is_rejected(self) -> None:
        pair = _pair(0, 1, self.points, self.points + (1000., 0.))
        with self.assertRaisesRegex(RuntimeError, r"Pixel validation.*Disconnected match graph"):
            estimate_global_transforms((pair,), _images(self.image, self.image.copy()), self.config)

    def test_small_verified_overlap_keeps_feature_fit_when_refinement_has_no_samples(self) -> None:
        image = np.full((48, 48, 3), 245, dtype=np.uint8)
        cv.rectangle(image, (8, 8), (38, 38), (20, 20, 20), 2)
        cv.line(image, (8, 8), (38, 38), (20, 20, 20), 2)
        points = np.array(((10., 10.), (35., 10.), (10., 35.), (35., 35.)))
        transforms = estimate_global_transforms((_pair(0, 1, points, points.copy()),), _images(image, image.copy()), self.config)
        np.testing.assert_allclose(transforms, (np.eye(3), np.eye(3)), atol=1e-12)

    def test_actual_corner_correction_is_bounded(self) -> None:
        images = _images(self.image, self.image.copy())
        pair = _pair(0, 1, self.points, self.points + (3., 0.))
        feature = estimate_global_transforms((pair,), images, StitchingConfig(pixel_refinement=False))
        pyramids = prepare_pixel_images(images)
        config = StitchingConfig(pixel_max_correction=1., pixel_min_correlation=.05, pixel_max_samples=6000)
        refined = refine_global_pixels(feature, (pair,), pyramids, config)
        movement = _maximum_pose_movement(feature, refined, pyramids)
        self.assertLessEqual(movement, 1.000001)
        self.assertGreater(movement, .5)

    def test_analytic_jacobian_for_both_moving_images(self) -> None:
        images = _images(self.image, self.image.copy(), self.image.copy())
        pair = _pair(1, 2, self.points, self.points.copy())
        pyramids = prepare_pixel_images(images)
        base = (np.eye(3), np.eye(3), np.eye(3))
        samples = _make_samples((pair,), pyramids, base, 0, self.config)
        centres = np.array(((319.5, 239.5),) * 3)
        objective = _PixelObjective(base, samples, centres, np.linalg.norm(centres, axis=1), 20.)
        parameters = np.array((.7, .32, -.27, -.4, -.21, .46))
        analytic = objective.jacobian(parameters).toarray()
        finite = np.empty_like(analytic)
        for column in range(len(parameters)):
            delta = np.zeros_like(parameters)
            delta[column] = 1e-5
            finite[:, column] = (objective.residuals(parameters + delta) - objective.residuals(parameters - delta)) / 2e-5
        np.testing.assert_allclose(analytic, finite, atol=1e-7, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
