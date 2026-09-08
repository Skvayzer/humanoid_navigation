#!/usr/bin/env python3

import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import global_localization_query as glq  # noqa: E402


def write_occupancy_map(directory: Path) -> Path:
    # 60x60 grid at 0.5 m; an L-shaped wall pattern with a distinctive corner.
    grid = np.zeros((60, 60), dtype=bool)
    grid[20, 10:50] = True
    grid[20:55, 10] = True
    grid[40, 25:45] = True
    # PGM rows are top-down; the loader flips them back up.
    pixels = np.where(np.flipud(grid), 0, 254).astype(np.uint8)
    image_path = directory / "map.pgm"
    Image.fromarray(pixels, mode="L").save(image_path)
    yaml_path = directory / "map.yaml"
    yaml_path.write_text(
        'image: "map.pgm"\n'
        "resolution: 0.5\n"
        "origin: [-10.0, -5.0, 0.0]\n"
        "negate: 0\n"
        "occupied_thresh: 0.65\n"
        "free_thresh: 0.196\n")
    return yaml_path


def make_scan(grid_yaml: Path, true_x: float, true_y: float, true_yaw: float):
    # Sample world points on the walls near the true pose and express them in
    # the scan frame, inside the z band and outside the min-range filter.
    engine_map = glq.bbs_engine.load_occupancy_map(grid_yaml)
    occupied = np.argwhere(engine_map.occupied)
    world_x = engine_map.origin_x_m + (occupied[:, 1] + 0.5) * engine_map.resolution_m
    world_y = engine_map.origin_y_m + (occupied[:, 0] + 0.5) * engine_map.resolution_m
    near = (np.abs(world_x - true_x) < 15.0) & (np.abs(world_y - true_y) < 15.0)
    world = np.stack([world_x[near], world_y[near]], axis=1)

    c = math.cos(-true_yaw)
    s = math.sin(-true_yaw)
    dx = world[:, 0] - true_x
    dy = world[:, 1] - true_y
    scan_x = c * dx - s * dy
    scan_y = s * dx + c * dy
    keep = np.hypot(scan_x, scan_y) >= 1.5
    points = np.stack(
        [scan_x[keep], scan_y[keep], np.full(int(keep.sum()), 1.0)], axis=1)
    clutter = np.array([[0.1, 0.0, 1.0], [3.0, 1.0, 9.0], [2.0, -1.0, -3.0]])
    return np.vstack([points, clutter])


def test_query_recovers_known_pose():
    with tempfile.TemporaryDirectory() as tmp:
        yaml_path = write_occupancy_map(Path(tmp))
        true_x, true_y, true_yaw = -1.0, 7.0, math.radians(30.0)
        config = glq.GlobalLocalizationConfig(
            angular_resolution_rad=math.radians(15.0),
            max_candidates=8,
            min_range_m=1.0,
        )
        engine = glq.GlobalLocalizationEngine(yaml_path, config)
        result = engine.query(make_scan(yaml_path, true_x, true_y, true_yaw))

        assert result.candidates, "expected at least one candidate"
        assert len(result.candidates) <= 8
        top = result.candidates[0]
        assert math.hypot(top.x_m - true_x, top.y_m - true_y) <= 1.0, (
            top.x_m, top.y_m)
        yaw_error = abs(
            glq.bbs_engine.normalize_angle_rad(top.yaw_rad - true_yaw))
        assert yaw_error <= math.radians(15.0) + 1.0e-9, yaw_error
        scores = [candidate.score for candidate in result.candidates]
        assert scores == sorted(scores, reverse=True)


def test_query_handles_empty_scan():
    with tempfile.TemporaryDirectory() as tmp:
        yaml_path = write_occupancy_map(Path(tmp))
        engine = glq.GlobalLocalizationEngine(
            yaml_path, glq.GlobalLocalizationConfig())
        result = engine.query(np.empty((0, 3), dtype=np.float64))
        assert result.candidates == []
        assert result.scan_point_count == 0


def test_matching_grid_is_coarsened_to_native_resolution():
    with tempfile.TemporaryDirectory() as tmp:
        yaml_path = write_occupancy_map(Path(tmp))
        engine = glq.GlobalLocalizationEngine(
            yaml_path,
            glq.GlobalLocalizationConfig(matching_resolution_m=1.0),
        )

        assert engine.occupancy_map.resolution_m == 1.0
        assert engine.occupancy_map.occupied.shape == (30, 30)
        assert engine.occupancy_map.occupied.any()


def test_scan_age_rejects_mixed_sensor_and_epoch_clock_domains():
    assert glq.scan_age_sec(100.0, 99.75) == 0.25
    assert glq.scan_age_sec(99.75, 100.0) == 0.0
    assert glq.scan_age_sec(1_700_000_000.0, 18_505.0) is None


def test_query_rejects_candidates_below_bbs_threshold():
    with tempfile.TemporaryDirectory() as tmp:
        yaml_path = write_occupancy_map(Path(tmp))
        true_x, true_y, true_yaw = -1.0, 7.0, math.radians(30.0)
        engine = glq.GlobalLocalizationEngine(
            yaml_path,
            glq.GlobalLocalizationConfig(
                angular_resolution_rad=math.radians(15.0),
                max_candidates=8,
                min_range_m=1.0,
                min_bbs_score=1.01,
            ),
        )
        result = engine.query(make_scan(yaml_path, true_x, true_y, true_yaw))

        assert result.candidates == []
        assert result.scan_point_count > 0


def test_bbs_cpp_search_path_includes_installed_lib_dir():
    with tempfile.TemporaryDirectory() as tmp:
        prefix = Path(tmp) / "install"
        module_dir = prefix / "lib" / "lidar_localization_ros2"
        module_dir.mkdir(parents=True)
        old_ament = os.environ.get("AMENT_PREFIX_PATH")
        old_path = list(sys.path)
        try:
            os.environ["AMENT_PREFIX_PATH"] = str(prefix)
            glq._append_module_dirs("bbs_cpp")
            assert str(module_dir) in sys.path
        finally:
            if old_ament is None:
                os.environ.pop("AMENT_PREFIX_PATH", None)
            else:
                os.environ["AMENT_PREFIX_PATH"] = old_ament
            sys.path[:] = old_path


class _FakeScoreResult:
    def __init__(
        self,
        fitness,
        converged,
        refined_x,
        refined_y,
        refined_z,
        refined_yaw,
        refined_roll=0.0,
        refined_pitch=0.0,
    ):
        self.fitness = fitness
        self.converged = converged
        self.refined_x = refined_x
        self.refined_y = refined_y
        self.refined_z = refined_z
        self.refined_roll = refined_roll
        self.refined_pitch = refined_pitch
        self.refined_yaw = refined_yaw
        self.target_point_count = 1000
        self.source_point_count = 512


class _FakeRegistrationScorer:
    def __init__(self, results):
        self._results = results

    def score_candidate(self, scan_xyz, x, y, z, yaw):
        return self._results[(x, y, z, yaw)]


class _FakeFinalRegistrationScorer:
    def __init__(self, results):
        self._results = results
        self.calls = []

    def refine_candidate(self, scan_xyz, x, y, z, roll, pitch, yaw):
        key = (x, y, z, roll, pitch, yaw)
        self.calls.append(key)
        return self._results[key]


def test_score_with_registration_rewrites_converged_candidate_pose():
    with tempfile.TemporaryDirectory() as tmp:
        yaml_path = write_occupancy_map(Path(tmp))
        engine = glq.GlobalLocalizationEngine(
            yaml_path,
            glq.GlobalLocalizationConfig(registration_refine_candidates=True))
        engine.registration_scorer = _FakeRegistrationScorer({
            (-13.5, 29.7, 0.0, math.radians(10.0)): _FakeScoreResult(
                fitness=0.042,
                converged=True,
                refined_x=-10.5,
                refined_y=27.1,
                refined_z=1.2,
                refined_roll=math.radians(3.0),
                refined_pitch=math.radians(-4.0),
                refined_yaw=math.radians(12.0),
            ),
            (5.0, 6.0, 0.0, math.radians(20.0)): _FakeScoreResult(
                fitness=float("inf"),
                converged=False,
                refined_x=99.0,
                refined_y=99.0,
                refined_z=99.0,
                refined_yaw=math.radians(99.0),
            ),
        })
        raw_candidates = [
            glq.GlobalLocalizationCandidate(
                x_m=-13.5,
                y_m=29.7,
                z_m=0.0,
                yaw_rad=math.radians(10.0),
                score=0.99,
                hit_count=100,
                point_count=512,
                bbs_score=0.99,
            ),
            glq.GlobalLocalizationCandidate(
                x_m=5.0,
                y_m=6.0,
                z_m=0.0,
                yaw_rad=math.radians(20.0),
                score=0.95,
                hit_count=90,
                point_count=512,
                bbs_score=0.95,
            ),
        ]
        ranked = engine._score_with_registration(
            np.zeros((8, 3), dtype=np.float64), raw_candidates)

        assert ranked[0].x_m == -10.5
        assert ranked[0].y_m == 27.1
        assert ranked[0].z_m == 1.2
        assert ranked[0].roll_rad == math.radians(3.0)
        assert ranked[0].pitch_rad == math.radians(-4.0)
        assert ranked[0].yaw_rad == math.radians(12.0)
        assert ranked[0].bbs_score == 0.99
        assert ranked[0].registration_fitness == 0.042

        unconverged = next(
            candidate for candidate in ranked
            if candidate.x_m == 5.0 and candidate.y_m == 6.0)
        assert unconverged.yaw_rad == math.radians(20.0)
        assert unconverged.z_m == 0.0
        assert not unconverged.registration_converged


def test_score_with_registration_keeps_raw_pose_by_default():
    with tempfile.TemporaryDirectory() as tmp:
        yaml_path = write_occupancy_map(Path(tmp))
        engine = glq.GlobalLocalizationEngine(
            yaml_path, glq.GlobalLocalizationConfig())
        engine.registration_scorer = _FakeRegistrationScorer({
            (-13.5, 29.7, 0.0, math.radians(10.0)): _FakeScoreResult(
                fitness=0.042,
                converged=True,
                refined_x=-10.5,
                refined_y=27.1,
                refined_z=1.2,
                refined_yaw=math.radians(12.0),
            ),
        })
        raw_candidates = [
            glq.GlobalLocalizationCandidate(
                x_m=-13.5,
                y_m=29.7,
                z_m=0.0,
                yaw_rad=math.radians(10.0),
                score=0.99,
                hit_count=100,
                point_count=512,
                bbs_score=0.99,
            ),
        ]
        ranked = engine._score_with_registration(
            np.zeros((8, 3), dtype=np.float64), raw_candidates)

        assert ranked[0].x_m == -13.5
        assert ranked[0].y_m == 29.7
        assert ranked[0].yaw_rad == math.radians(10.0)
        assert ranked[0].registration_fitness == 0.042


def test_final_registration_uses_full_pose_and_only_returns_refined_candidates():
    with tempfile.TemporaryDirectory() as tmp:
        yaml_path = write_occupancy_map(Path(tmp))
        engine = glq.GlobalLocalizationEngine(
            yaml_path,
            glq.GlobalLocalizationConfig(final_refinement_candidate_count=1),
        )
        seed = (-13.5, 29.7, 1.1, 0.2, -0.3, math.radians(10.0))
        scorer = _FakeFinalRegistrationScorer({
            seed: _FakeScoreResult(
                fitness=0.031,
                converged=True,
                refined_x=-13.2,
                refined_y=29.4,
                refined_z=1.3,
                refined_roll=0.21,
                refined_pitch=-0.28,
                refined_yaw=math.radians(11.0),
            ),
        })
        engine.final_registration_scorer = scorer
        candidates = [
            glq.GlobalLocalizationCandidate(
                x_m=seed[0], y_m=seed[1], z_m=seed[2],
                roll_rad=seed[3], pitch_rad=seed[4], yaw_rad=seed[5],
                score=0.9, hit_count=100, point_count=512,
                bbs_score=0.9, registration_fitness=0.1,
                registration_converged=True,
            ),
            glq.GlobalLocalizationCandidate(
                x_m=5.0, y_m=6.0, z_m=0.0, yaw_rad=0.0,
                score=0.8, hit_count=90, point_count=512,
                bbs_score=0.8, registration_fitness=0.2,
                registration_converged=True,
            ),
        ]

        ranked = engine._refine_with_final_registration(
            np.zeros((8, 3), dtype=np.float64), candidates)

        assert scorer.calls == [seed]
        assert len(ranked) == 1
        assert ranked[0].x_m == -13.2
        assert ranked[0].y_m == 29.4
        assert ranked[0].z_m == 1.3
        assert ranked[0].roll_rad == 0.21
        assert ranked[0].pitch_rad == -0.28
        assert ranked[0].yaw_rad == math.radians(11.0)
        assert ranked[0].registration_fitness == 0.031


if __name__ == "__main__":
    test_query_recovers_known_pose()
    test_query_handles_empty_scan()
    test_matching_grid_is_coarsened_to_native_resolution()
    test_scan_age_rejects_mixed_sensor_and_epoch_clock_domains()
    test_query_rejects_candidates_below_bbs_threshold()
    test_bbs_cpp_search_path_includes_installed_lib_dir()
    test_score_with_registration_rewrites_converged_candidate_pose()
    test_score_with_registration_keeps_raw_pose_by_default()
    test_final_registration_uses_full_pose_and_only_returns_refined_candidates()
    print("test_global_localization_query: all tests passed")
