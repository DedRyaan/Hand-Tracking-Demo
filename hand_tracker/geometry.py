"""Geometry and movement calculations for tracked hand landmarks."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .constants import BONE_CONNECTIONS, FALLBACK_HAND_FACES, FINGER_JOINT_TRIPLETS


@dataclass
class BoneMeasurement:
    name: str
    start: int
    end: int
    vector: np.ndarray
    direction: np.ndarray
    length: float


def _safe_norm(vector: np.ndarray) -> tuple[np.ndarray, float]:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        return np.zeros_like(vector), 0.0
    return vector / norm, norm


def joint_angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Returns angle ABC in degrees."""
    ba = a - b
    bc = c - b

    ba_norm = float(np.linalg.norm(ba))
    bc_norm = float(np.linalg.norm(bc))

    if ba_norm < 1e-8 or bc_norm < 1e-8:
        return 180.0

    cosine = float(np.dot(ba, bc) / (ba_norm * bc_norm))
    cosine = float(np.clip(cosine, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def compute_finger_flexion(landmarks: np.ndarray) -> dict[str, float]:
    """Computes per-finger flexion where 0 is straight, larger is more bent."""
    flexion: dict[str, float] = {}

    for finger_name, triplets in FINGER_JOINT_TRIPLETS.items():
        bends: list[float] = []
        for a_idx, b_idx, c_idx in triplets:
            angle = joint_angle_deg(landmarks[a_idx], landmarks[b_idx], landmarks[c_idx])
            bends.append(max(0.0, 180.0 - angle))
        flexion[finger_name] = float(np.mean(bends)) if bends else 0.0

    return flexion


def finger_state(flexion: float) -> str:
    if flexion < 25.0:
        return "Open"
    if flexion < 65.0:
        return "Curved"
    return "Closed"


def compute_bone_measurements(landmarks: np.ndarray) -> list[BoneMeasurement]:
    """Computes direction and length for each hand bone connection."""
    output: list[BoneMeasurement] = []

    for start, end in BONE_CONNECTIONS:
        vector = landmarks[end] - landmarks[start]
        direction, length = _safe_norm(vector)
        output.append(
            BoneMeasurement(
                name=f"{start}->{end}",
                start=start,
                end=end,
                vector=vector,
                direction=direction,
                length=length,
            )
        )

    return output


def generate_mesh_faces(landmarks: np.ndarray) -> np.ndarray:
    """Generates triangles over hand landmarks using Delaunay triangulation."""
    if landmarks.shape[0] < 3:
        return np.asarray(FALLBACK_HAND_FACES, dtype=np.int32)

    points_2d = landmarks[:, :2].astype(np.float32)

    min_xy = points_2d.min(axis=0)
    max_xy = points_2d.max(axis=0)
    span = np.maximum(max_xy - min_xy, 1e-6)
    normalized = (points_2d - min_xy) / span

    scale = 1000.0
    points = normalized * scale

    subdiv = cv2.Subdiv2D((0, 0, int(scale), int(scale)))
    inserted: list[tuple[float, float]] = []

    for point in points:
        x = float(np.clip(point[0], 1.0, scale - 2.0))
        y = float(np.clip(point[1], 1.0, scale - 2.0))
        subdiv.insert((x, y))
        inserted.append((x, y))

    triangles = subdiv.getTriangleList()
    if triangles is None or len(triangles) == 0:
        return np.asarray(FALLBACK_HAND_FACES, dtype=np.int32)

    inserted_points = np.asarray(inserted, dtype=np.float32)
    faces: set[tuple[int, int, int]] = set()

    for tri in triangles:
        tri_points = np.asarray(
            [[tri[0], tri[1]], [tri[2], tri[3]], [tri[4], tri[5]]],
            dtype=np.float32,
        )

        indices: list[int] = []
        valid = True

        for vertex in tri_points:
            diff = inserted_points - vertex
            dist_sq = np.sum(diff * diff, axis=1)
            idx = int(np.argmin(dist_sq))
            if float(dist_sq[idx]) > 6.0:
                valid = False
                break
            indices.append(idx)

        if not valid or len(set(indices)) < 3:
            continue

        tri_world = landmarks[np.asarray(indices)]
        edge_lengths = [
            float(np.linalg.norm(tri_world[1] - tri_world[0])),
            float(np.linalg.norm(tri_world[2] - tri_world[1])),
            float(np.linalg.norm(tri_world[0] - tri_world[2])),
        ]

        if max(edge_lengths) > 0.45:
            continue

        faces.add(tuple(sorted(indices)))

    if len(faces) < 8:
        return np.asarray(FALLBACK_HAND_FACES, dtype=np.int32)

    return np.asarray(sorted(faces), dtype=np.int32)


def normalize_landmarks_for_view(landmarks: np.ndarray) -> np.ndarray:
    """Converts normalized MediaPipe landmarks into centered 3D view coordinates."""
    x = (landmarks[:, 0] - 0.5) * 2.0
    y = -(landmarks[:, 1] - 0.5) * 2.0
    z = -landmarks[:, 2] * 2.0
    return np.column_stack((x, y, z)).astype(np.float32)
