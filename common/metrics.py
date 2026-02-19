from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class EpisodeMetricTracker:
    """Tracks NC/DAC/TTC/COM/RC metrics within a single episode."""

    dac_threshold: float = 1.5
    ttc_threshold: float = 5.0
    accel_threshold: float = 2.0
    yaw_acc_threshold: float = 0.5

    collision: bool = False
    dac_total: int = 0
    dac_hits: int = 0
    ttc_flags: List[float] = field(default_factory=list)
    com_total: int = 0
    com_hits: int = 0

    agent_start: Optional[np.ndarray] = None
    agent_end: Optional[np.ndarray] = None
    expert_start: Optional[np.ndarray] = None
    expert_end: Optional[np.ndarray] = None
    expert_path_points: Optional[np.ndarray] = None
    expert_path_cumlen: Optional[np.ndarray] = None
    expert_total_length: Optional[float] = None

    _prev_ang_vel: Optional[float] = None
    _prev_timestamp: Optional[float] = None

    _last_seg_idx: Optional[int] = None
    _last_progress: float = 0.0
    _off_route_tol: float = 5.0
    _back_slop: float = 2.0 

    def update(self, info: Dict[str, Any]) -> None:
        """Update metric accumulators with the latest env info dict."""
        if not isinstance(info, dict):
            return
        self.collision |= bool(info.get('collision', 0))
        self._update_dac(info)
        self._update_ttc(info)
        self._update_comfort(info)
        self._update_route(info)

    def finalize(self) -> Dict[str, float]:
        """Return final per-episode metrics."""
        dac_score = self.dac_hits / self.dac_total if self.dac_total > 0 else 0.0
        ttc_score = float(np.mean(self.ttc_flags)) if self.ttc_flags else 1.0
        com_score = self.com_hits / self.com_total if self.com_total > 0 else 0.0
        rc_score = self._compute_route_completion()

        return {
            'NC': 0.0 if self.collision else 1.0,
            'DAC': dac_score,
            'TTC': ttc_score,
            'COM': com_score,
            'RC': rc_score,
        }

    def _update_dac(self, info: Dict[str, Any]) -> None:
        deviation = info.get('position_deviation')
        if deviation is None:
            return
        self.dac_total += 1
        if float(deviation) <= self.dac_threshold:
            self.dac_hits += 1

    def _update_ttc(self, info: Dict[str, Any]) -> None:
        ego_state = info.get('ego_state') or {}
        ego_vel = np.asarray(ego_state.get('ego_velocity', [0.0, 0.0]), dtype=np.float32)
        surroundings = info.get('surroundings') or []

        min_ttc = np.inf
        for obj in surroundings:
            obj_pos = np.asarray(obj.get('position', []), dtype=np.float32)
            obj_vel = np.asarray(obj.get('velocity', []), dtype=np.float32)
            if obj_pos.shape[0] < 2 or obj_vel.shape[0] < 2:
                continue
            rel_pos = obj_pos
            dist = np.linalg.norm(rel_pos)
            if dist < 1e-3:
                min_ttc = 0.0
                break
            rel_vel = obj_vel - ego_vel
            rel_dir = rel_pos / dist
            closing_speed = -float(np.dot(rel_vel, rel_dir))
            if closing_speed <= 1e-3:
                continue
            ttc = dist / closing_speed
            if ttc < min_ttc:
                min_ttc = ttc

        if np.isfinite(min_ttc):
            self.ttc_flags.append(1.0 if min_ttc >= self.ttc_threshold else 0.0)
        else:
            # No approaching objects -> treat as safe timestep.
            self.ttc_flags.append(1.0)

    def _update_comfort(self, info: Dict[str, Any]) -> None:
        ego_state = info.get('ego_state') or {}
        acc = np.asarray(ego_state.get('ego_acceleration', [0.0, 0.0]), dtype=np.float32)
        lin_acc = float(np.linalg.norm(acc))
        angular_vel = float(ego_state.get('angular_velocity', 0.0) or 0.0)
        timestamp = float(ego_state.get('timestamp', 0.0) or 0.0)

        ang_acc = 0.0
        if self._prev_ang_vel is not None and self._prev_timestamp is not None:
            dt = max(timestamp - self._prev_timestamp, 1e-3)
            ang_acc = (angular_vel - self._prev_ang_vel) / dt

        self._prev_ang_vel = angular_vel
        self._prev_timestamp = timestamp

        comfortable = (lin_acc <= self.accel_threshold) and (abs(ang_acc) <= self.yaw_acc_threshold)
        self.com_total += 1
        if comfortable:
            self.com_hits += 1

    def _update_route(self, info: Dict[str, Any]) -> None:
        ego_state = info.get('ego_state') or {}
        position = ego_state.get('position_world')
        if position is not None:
            pos_arr = np.asarray(position, dtype=np.float32)
            if self.agent_start is None:
                self.agent_start = pos_arr
            self.agent_end = pos_arr

        path_info = info.get('expert_path')
        if isinstance(path_info, dict):
            start_xy = path_info.get('start_xy')
            end_xy = path_info.get('end_xy')
            if self.expert_start is None and start_xy is not None:
                self.expert_start = np.asarray(start_xy, dtype=np.float32)
            if end_xy is not None:
                self.expert_end = np.asarray(end_xy, dtype=np.float32)

            path_xy = path_info.get('path_xy')
            if path_xy is not None:
                pts = np.asarray(path_xy, dtype=np.float32)
                if pts.ndim == 2 and pts.shape[0] >= 2:
                    self.expert_path_points = pts
                    cumlen = path_info.get('path_cumlen')
                    if cumlen is not None:
                        cum_arr = np.asarray(cumlen, dtype=np.float32)
                        if cum_arr.shape[0] == pts.shape[0]:
                            self.expert_path_cumlen = cum_arr
                        else:
                            self.expert_path_cumlen = self._compute_cumulative_lengths(pts)
                    else:
                        self.expert_path_cumlen = self._compute_cumulative_lengths(pts)
                    if self.expert_path_cumlen is not None and self.expert_path_cumlen.size > 0:
                        self.expert_total_length = float(self.expert_path_cumlen[-1])
                        self._last_seg_idx = None

    def _compute_route_completion(self) -> float:
        if self.agent_end is not None and self.expert_total_length is not None and self.expert_total_length > 1e-6:
            progress = self._project_progress_along_path(self.agent_end)
            if progress is not None:
                ratio = np.clip(progress / self.expert_total_length, 0.0, 1.0)
                return float(ratio)

        if (
            self.agent_start is None or
            self.agent_end is None or
            self.expert_start is None or
            self.expert_end is None
        ):
            return 0.0
        agent_disp = float(np.linalg.norm(self.agent_end - self.agent_start))
        expert_disp = float(np.linalg.norm(self.expert_end - self.expert_start))
        if expert_disp < 1e-3:
            return 0.0
        return float(np.clip(agent_disp / expert_disp, 0.0, 1.0))

    @staticmethod
    def _compute_cumulative_lengths(points: np.ndarray) -> Optional[np.ndarray]:
        if points.shape[0] < 2:
            return None
        seg_lengths = np.linalg.norm(points[1:] - points[:-1], axis=1)
        cumlen = np.concatenate(([0.0], np.cumsum(seg_lengths)))
        return cumlen.astype(np.float32)

    def _project_progress_along_path(self, position: np.ndarray) -> Optional[float]:
        if (
            self.expert_path_points is None or
            self.expert_path_points.shape[0] < 2 or
            self.expert_path_cumlen is None
        ):
            return None

        pts = self.expert_path_points
        cumlen = self.expert_path_cumlen
        N = len(pts) - 1
        if self._last_seg_idx is None:
            seg_range = range(0, N)
        else:
            k = 15
            lo = max(0, self._last_seg_idx - k)
            hi = min(N, self._last_seg_idx + k)
            seg_range = range(lo, hi)

        best_progress = None
        best_dist = np.inf
        best_idx = None

        for idx in seg_range:
            p0 = pts[idx]
            p1 = pts[idx + 1]
            seg = p1 - p0
            seg_len = np.linalg.norm(seg)
            if seg_len < 1e-6:
                continue
            t = float(np.dot(position - p0, seg) / (seg_len ** 2))
            t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t
            proj = p0 + t * seg
            dist = np.linalg.norm(position - proj)
            if dist < best_dist:
                base = float(cumlen[idx]) if idx < len(cumlen) else 0.0
                best_progress = base + t * seg_len
                best_dist = dist
                best_idx = idx

        if best_progress is None:
            return None
        if best_dist > self._off_route_tol:
            return self._last_progress

        if best_progress + self._back_slop < self._last_progress:
            best_progress = self._last_progress - self._back_slop

        self._last_progress = best_progress
        self._last_seg_idx = best_idx
        return best_progress

