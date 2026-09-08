#!/usr/bin/env python3
"""ROS-free global localization query logic for the G2 runtime service.

Wraps the BBS_2D engine from make_bbs_relocalization_attempts with occupancy
map loading, scan preparation, and candidate-to-world-pose conversion, so the
runtime node and offline tooling share one code path and the logic stays unit
testable without rclpy.
"""

import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import g2_candidate_registration_rank_policy as g2_rank  # noqa: E402
import make_bbs_relocalization_attempts as bbs_engine  # noqa: E402


def _candidate_module_dirs(module_name: str) -> List[Path]:
    """Return likely directories for an optional installed pybind11 module."""
    dirs = [
        Path(__file__).resolve().parent,
        Path(sys.argv[0]).absolute().parent,
    ]
    for prefix in os.environ.get("AMENT_PREFIX_PATH", "").split(os.pathsep):
        if prefix:
            dirs.append(Path(prefix) / "lib" / "lidar_localization_ros2")
    return dirs


def _append_module_dirs(module_name: str) -> None:
    for directory in _candidate_module_dirs(module_name):
        if directory.is_dir():
            text = str(directory)
            if text not in sys.path:
                sys.path.insert(0, text)


@dataclass(frozen=True)
class GlobalLocalizationConfig:
    z_min_m: float = 0.5
    z_max_m: float = 5.0
    min_range_m: float = 1.0
    max_scan_points: int = 512
    angular_resolution_rad: float = math.radians(5.0)
    pyramid_depth: int = 4
    matching_resolution_m: float = 0.0
    max_candidates: int = 16
    min_bbs_score: float = 0.0
    nms_radius_m: float = 2.0
    dilate_cells: int = 1
    seed_z_m: float = 0.0
    registration_seed_z_m: float = 0.0
    # Opt-in: use the compiled C++ branch-and-bound backend (bbs_cpp) when it is
    # importable. Defaults to False so a stock checkout runs pure Python; if the
    # module is missing the engine logs once and silently falls back to Python.
    use_cpp_backend: bool = False
    # Opt-in: re-rank BBS candidates by NDT registration fitness against map_path.
    enable_registration_scoring: bool = False
    # Opt-in: report the NDT-refined pose of a converged candidate instead of the raw
    # BBS cell pose. Helps when candidates are coarse (a converged candidate can be
    # metres off while its refinement is exact), but on high-aliasing maps refinement
    # also snaps *wrong* hypotheses to locally-perfect alignments and collapses nearby
    # walk candidates onto one pose — keep it off unless the scenario shows it helps.
    registration_refine_candidates: bool = False
    map_path: Optional[str] = None
    registration_score_gate: float = 6.0
    ndt_resolution: float = 1.0
    ndt_step_size: float = 0.1
    ndt_transform_epsilon: float = 0.01
    ndt_max_iterations: int = 30
    ndt_num_threads: int = 1
    ndt_scan_voxel_leaf_size: float = 1.0
    ndt_target_voxel_leaf_size: float = 0.2
    ndt_local_map_radius: float = 150.0
    ndt_min_target_points: int = 100
    enable_final_registration_refinement: bool = False
    final_refinement_candidate_count: int = 4
    final_ndt_scan_voxel_leaf_size: float = 0.25
    final_ndt_target_voxel_leaf_size: float = 0.25


@dataclass(frozen=True)
class GlobalLocalizationCandidate:
    x_m: float
    y_m: float
    z_m: float
    yaw_rad: float
    score: float
    hit_count: int
    point_count: int
    bbs_score: float
    registration_fitness: Optional[float] = None
    registration_converged: Optional[bool] = None
    roll_rad: float = 0.0
    pitch_rad: float = 0.0


@dataclass(frozen=True)
class GlobalLocalizationResult:
    candidates: List[GlobalLocalizationCandidate]
    scan_point_count: int
    registration_scoring_enabled: bool = False
    registration_scoring_backend: Optional[str] = None
    registration_scoring_error: Optional[str] = None


def _coarsen_occupancy_map(occupancy_map, target_resolution_m: float):
    source_resolution_m = float(occupancy_map.resolution_m)
    if target_resolution_m < source_resolution_m:
        raise ValueError(
            "matching_resolution_m cannot be finer than the occupancy map"
        )
    ratio = target_resolution_m / source_resolution_m
    factor = int(round(ratio))
    if factor < 1 or not math.isclose(ratio, factor, rel_tol=0.0, abs_tol=1.0e-6):
        raise ValueError(
            "matching_resolution_m must be an integer multiple of map resolution"
        )
    if factor == 1:
        return occupancy_map

    occupied = occupancy_map.occupied
    padded_height = int(math.ceil(occupied.shape[0] / factor) * factor)
    padded_width = int(math.ceil(occupied.shape[1] / factor) * factor)
    padded = np.zeros((padded_height, padded_width), dtype=bool)
    padded[: occupied.shape[0], : occupied.shape[1]] = occupied
    pooled = padded.reshape(
        padded_height // factor, factor, padded_width // factor, factor
    ).any(axis=(1, 3))
    return bbs_engine.OccupancyMap(
        occupied=pooled,
        resolution_m=source_resolution_m * factor,
        origin_x_m=occupancy_map.origin_x_m,
        origin_y_m=occupancy_map.origin_y_m,
        origin_yaw_rad=occupancy_map.origin_yaw_rad,
    )


def scan_age_sec(
    now_sec: float,
    scan_stamp_sec: float,
    max_same_clock_delta_sec: float = 3600.0,
):
    """Return scan age only when both timestamps plausibly share one clock."""
    if now_sec <= 0.0 or scan_stamp_sec <= 0.0:
        return None
    delta_sec = now_sec - scan_stamp_sec
    if abs(delta_sec) > max_same_clock_delta_sec:
        return None
    return max(0.0, delta_sec)


def _candidate_from_ranked(ranked: g2_rank.RankedG2Candidate) -> GlobalLocalizationCandidate:
    return GlobalLocalizationCandidate(
        x_m=ranked.x_m,
        y_m=ranked.y_m,
        z_m=ranked.z_m,
        yaw_rad=ranked.yaw_rad,
        score=ranked.score,
        hit_count=ranked.hit_count,
        point_count=ranked.point_count,
        bbs_score=ranked.bbs_score,
        registration_fitness=ranked.registration_fitness,
        registration_converged=ranked.registration_converged,
        roll_rad=ranked.roll_rad,
        pitch_rad=ranked.pitch_rad,
    )


class GlobalLocalizationEngine:
    """Loads the occupancy map once and answers repeated scan queries."""

    def __init__(self, occupancy_yaml, config):
        self.config = config
        self.occupancy_map = bbs_engine.load_occupancy_map(Path(occupancy_yaml))
        if config.matching_resolution_m > 0.0:
            self.occupancy_map = _coarsen_occupancy_map(
                self.occupancy_map, config.matching_resolution_m)
        matching_grid = self.occupancy_map.occupied
        for _ in range(max(0, config.dilate_cells)):
            matching_grid = bbs_engine._dilate_one_cell(matching_grid)
        self.matching_grid = matching_grid

        # Resolve the search backend once. The C++ module (bbs_cpp) is bit-exact
        # to bbs_engine.branch_and_bound_candidates and exposes the same call
        # signature and candidate attributes, so query() is backend-agnostic.
        self.backend = "python"
        self.backend_error = None
        self._search = bbs_engine.branch_and_bound_candidates
        if config.use_cpp_backend:
            try:
                _append_module_dirs("bbs_cpp")
                import bbs_cpp  # noqa: E402
                self._search = bbs_cpp.branch_and_bound_candidates
                self.backend = "cpp"
            except ImportError as exc:  # pragma: no cover - depends on build
                self.backend_error = str(exc)

        self.registration_scorer = None
        self.final_registration_scorer = None
        self.registration_scoring_error = None
        self._registration_seed_z_m = config.registration_seed_z_m
        if config.enable_registration_scoring:
            if not config.map_path:
                self.registration_scoring_error = "map_path required for registration scoring"
            else:
                try:
                    _append_module_dirs("g2_ndt_score")
                    import g2_ndt_score  # noqa: E402
                    self.registration_scorer = g2_ndt_score.MapNdtScorer(
                        config.map_path,
                        config.ndt_resolution,
                        config.ndt_step_size,
                        config.ndt_transform_epsilon,
                        config.ndt_max_iterations,
                        config.ndt_num_threads,
                        config.ndt_scan_voxel_leaf_size,
                        config.ndt_target_voxel_leaf_size,
                        config.ndt_local_map_radius,
                        config.ndt_min_target_points,
                    )
                    if config.enable_final_registration_refinement:
                        self.final_registration_scorer = g2_ndt_score.MapNdtScorer(
                            config.map_path,
                            config.ndt_resolution,
                            config.ndt_step_size,
                            config.ndt_transform_epsilon,
                            config.ndt_max_iterations,
                            config.ndt_num_threads,
                            config.final_ndt_scan_voxel_leaf_size,
                            config.final_ndt_target_voxel_leaf_size,
                            config.ndt_local_map_radius,
                            config.ndt_min_target_points,
                        )
                except ImportError as exc:  # pragma: no cover - depends on build
                    self.registration_scoring_error = str(exc)

    def _score_with_registration(
        self,
        points_xyz,
        candidates: List[GlobalLocalizationCandidate],
    ) -> List[GlobalLocalizationCandidate]:
        if self.registration_scorer is None:
            return candidates

        ranked = []
        for candidate in candidates:
            score_z = (
                candidate.z_m
                if math.isfinite(candidate.z_m) and abs(candidate.z_m) > 1.0e-6
                else self._registration_seed_z_m)
            result = self.registration_scorer.score_candidate(
                points_xyz,
                candidate.x_m,
                candidate.y_m,
                score_z,
                candidate.yaw_rad,
            )
            fitness = (
                float(result.fitness)
                if result.converged and math.isfinite(result.fitness)
                else float("inf"))
            if self.config.registration_refine_candidates:
                pose_x, pose_y, pose_z, pose_roll, pose_pitch, pose_yaw = (
                    g2_rank.candidate_pose_from_registration(
                        g2_rank.RankedG2Candidate(
                            x_m=candidate.x_m,
                            y_m=candidate.y_m,
                            z_m=candidate.z_m,
                            yaw_rad=candidate.yaw_rad,
                            bbs_score=candidate.bbs_score,
                            registration_fitness=fitness,
                            score=candidate.score,
                            hit_count=candidate.hit_count,
                            point_count=candidate.point_count,
                            registration_converged=bool(result.converged),
                            roll_rad=candidate.roll_rad,
                            pitch_rad=candidate.pitch_rad,
                        ),
                        converged=bool(result.converged),
                        fitness=fitness if math.isfinite(fitness) else None,
                        refined_x=float(result.refined_x),
                        refined_y=float(result.refined_y),
                        refined_z=float(result.refined_z),
                        refined_roll=float(result.refined_roll),
                        refined_pitch=float(result.refined_pitch),
                        refined_yaw=float(result.refined_yaw),
                    )
                )
            else:
                pose_x, pose_y, pose_z, pose_roll, pose_pitch, pose_yaw = (
                    candidate.x_m,
                    candidate.y_m,
                    candidate.z_m,
                    candidate.roll_rad,
                    candidate.pitch_rad,
                    candidate.yaw_rad,
                )
            ranked.append(
                g2_rank.RankedG2Candidate(
                    x_m=pose_x,
                    y_m=pose_y,
                    z_m=pose_z,
                    yaw_rad=pose_yaw,
                    bbs_score=candidate.bbs_score,
                    registration_fitness=fitness,
                    score=candidate.score,
                    hit_count=candidate.hit_count,
                    point_count=candidate.point_count,
                    registration_converged=bool(result.converged),
                    roll_rad=pose_roll,
                    pitch_rad=pose_pitch,
                ))
        reranked = g2_rank.apply_registration_ranking(
            ranked, score_gate=self.config.registration_score_gate)
        return [_candidate_from_ranked(item) for item in reranked]

    def _refine_with_final_registration(
        self,
        points_xyz,
        candidates: List[GlobalLocalizationCandidate],
    ) -> List[GlobalLocalizationCandidate]:
        if self.final_registration_scorer is None:
            return candidates

        refinement_count = max(
            0, int(self.config.final_refinement_candidate_count))
        refined = []
        # Do not mix scores produced with different voxelizations. Recovery may
        # walk only the candidates that received this final registration pass.
        for candidate in candidates[:refinement_count]:
            result = self.final_registration_scorer.refine_candidate(
                points_xyz,
                candidate.x_m,
                candidate.y_m,
                candidate.z_m,
                candidate.roll_rad,
                candidate.pitch_rad,
                candidate.yaw_rad,
            )
            fitness = (
                float(result.fitness)
                if result.converged and math.isfinite(result.fitness)
                else float("inf")
            )
            pose = g2_rank.candidate_pose_from_registration(
                g2_rank.RankedG2Candidate(
                    x_m=candidate.x_m,
                    y_m=candidate.y_m,
                    z_m=candidate.z_m,
                    yaw_rad=candidate.yaw_rad,
                    bbs_score=candidate.bbs_score,
                    registration_fitness=fitness,
                    score=candidate.score,
                    hit_count=candidate.hit_count,
                    point_count=candidate.point_count,
                    registration_converged=bool(result.converged),
                    roll_rad=candidate.roll_rad,
                    pitch_rad=candidate.pitch_rad,
                ),
                converged=bool(result.converged),
                fitness=fitness if math.isfinite(fitness) else None,
                refined_x=float(result.refined_x),
                refined_y=float(result.refined_y),
                refined_z=float(result.refined_z),
                refined_roll=float(result.refined_roll),
                refined_pitch=float(result.refined_pitch),
                refined_yaw=float(result.refined_yaw),
            )
            x, y, z, roll, pitch, yaw = pose
            refined.append(
                g2_rank.RankedG2Candidate(
                    x_m=x,
                    y_m=y,
                    z_m=z,
                    yaw_rad=yaw,
                    bbs_score=candidate.bbs_score,
                    registration_fitness=fitness,
                    score=candidate.score,
                    hit_count=candidate.hit_count,
                    point_count=candidate.point_count,
                    registration_converged=bool(result.converged),
                    roll_rad=roll,
                    pitch_rad=pitch,
                )
            )
        reranked = g2_rank.apply_registration_ranking(
            refined, score_gate=self.config.registration_score_gate)
        return [_candidate_from_ranked(item) for item in reranked]

    def query(self, points_xyz):
        config = self.config
        resolution_m = self.occupancy_map.resolution_m
        scan_xy = bbs_engine.prepare_scan_xy(
            points_xyz,
            config.z_min_m,
            config.z_max_m,
            resolution_m,
            config.max_scan_points,
            min_range_m=config.min_range_m,
        )
        if scan_xy.shape[0] == 0:
            return GlobalLocalizationResult(
                candidates=[],
                scan_point_count=0,
                registration_scoring_enabled=self.registration_scorer is not None,
                registration_scoring_backend=(
                    "g2_ndt_score" if self.registration_scorer is not None else None),
                registration_scoring_error=self.registration_scoring_error,
            )

        grid_candidates = self._search(
            self.matching_grid,
            scan_xy,
            resolution_m,
            config.angular_resolution_rad,
            config.pyramid_depth,
            config.max_candidates,
            nms_radius_cells=int(
                round(max(0.0, config.nms_radius_m) / resolution_m)),
        )

        candidates = []
        for candidate in grid_candidates:
            if candidate.score < config.min_bbs_score:
                continue
            x_m, y_m = self.occupancy_map.grid_cell_center_to_world(
                candidate.tx_cell, candidate.ty_cell)
            yaw_rad = bbs_engine.normalize_angle_rad(
                self.occupancy_map.origin_yaw_rad + candidate.yaw_rad)
            bbs_score = candidate.score
            candidates.append(
                GlobalLocalizationCandidate(
                    x_m=x_m,
                    y_m=y_m,
                    z_m=config.seed_z_m,
                    yaw_rad=yaw_rad,
                    score=bbs_score,
                    hit_count=candidate.hit_count,
                    point_count=candidate.point_count,
                    bbs_score=bbs_score,
                ))
        if self.registration_scorer is not None:
            candidates = self._score_with_registration(points_xyz, candidates)
            candidates = self._refine_with_final_registration(
                points_xyz, candidates)

        return GlobalLocalizationResult(
            candidates=candidates,
            scan_point_count=int(scan_xy.shape[0]),
            registration_scoring_enabled=self.registration_scorer is not None,
            registration_scoring_backend=(
                "g2_ndt_score" if self.registration_scorer is not None else None),
            registration_scoring_error=self.registration_scoring_error,
        )
