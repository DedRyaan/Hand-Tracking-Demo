"""Hand tracking engine using MediaPipe Tasks hand landmarker."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

from .constants import BONE_CONNECTIONS, FINGER_ORDER, HAND_SIDE_ORDER
from .geometry import (
    BoneMeasurement,
    compute_bone_measurements,
    compute_finger_flexion,
    generate_mesh_faces,
)


DEFAULT_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)

PALM_CENTER_INDICES = np.asarray([0, 5, 9, 13, 17], dtype=np.int32)
PALM_WIDTH_PAIR = (5, 17)
DEPTH_OFFSET_SCALE = 1.15


@dataclass
class TrackerConfig:
    max_num_hands: int = 2
    min_hand_detection_confidence: float = 0.62
    min_hand_presence_confidence: float = 0.58
    min_tracking_confidence: float = 0.72
    inference_delegate: str = "auto"
    camera_fov_degrees: float = 62.0
    default_hand_width_m: float = 0.085
    min_depth_m: float = 0.12
    max_depth_m: float = 1.5
    smoothing_enabled: bool = True
    smoothing_slow_alpha: float = 0.32
    smoothing_fast_alpha: float = 0.82
    smoothing_velocity_scale: float = 0.018


@dataclass
class HandState:
    side: str
    handedness_score: float | None
    landmarks: np.ndarray
    world_landmarks: np.ndarray | None
    finger_flexion: dict[str, float]
    bones: list[BoneMeasurement]
    mesh_faces: np.ndarray
    camera_vertices: np.ndarray
    camera_center: np.ndarray
    estimated_depth_m: float


@dataclass
class HandFrameState:
    hands: list[HandState]


class AdaptiveLandmarkFilter:
    """Adaptive exponential smoother for responsive but stable landmarks."""

    def __init__(self, slow_alpha: float, fast_alpha: float, velocity_scale: float) -> None:
        self._slow_alpha = float(np.clip(slow_alpha, 0.01, 0.99))
        self._fast_alpha = float(np.clip(fast_alpha, self._slow_alpha, 0.99))
        self._velocity_scale = max(1e-6, float(velocity_scale))
        self._state: np.ndarray | None = None

    def reset(self) -> None:
        self._state = None

    def apply(self, landmarks: np.ndarray) -> np.ndarray:
        if self._state is None:
            self._state = landmarks.copy()
            return landmarks

        motion = float(np.mean(np.linalg.norm(landmarks - self._state, axis=1)))
        blend = float(np.clip(motion / self._velocity_scale, 0.0, 1.0))
        alpha = self._slow_alpha + (self._fast_alpha - self._slow_alpha) * blend

        filtered = alpha * landmarks + (1.0 - alpha) * self._state
        self._state = filtered
        return filtered


def estimate_depth_from_hand_width(
    landmarks: np.ndarray,
    frame_width_px: int,
    camera_fov_degrees: float,
    real_hand_width_m: float,
    min_depth_m: float,
    max_depth_m: float,
) -> float:
    """Estimate depth from apparent palm width using a pinhole approximation."""
    u0 = float(landmarks[PALM_WIDTH_PAIR[0], 0] * frame_width_px)
    u1 = float(landmarks[PALM_WIDTH_PAIR[1], 0] * frame_width_px)
    pixel_width = abs(u1 - u0)

    if pixel_width < 3.0:
        return float((min_depth_m + max_depth_m) * 0.5)

    fov_rad = np.radians(float(np.clip(camera_fov_degrees, 20.0, 150.0)))
    focal_px = frame_width_px / (2.0 * np.tan(fov_rad * 0.5))
    depth = (focal_px * real_hand_width_m) / pixel_width
    return float(np.clip(depth, min_depth_m, max_depth_m))


def normalized_to_camera_space(
    landmarks: np.ndarray,
    frame_width_px: int,
    frame_height_px: int,
    depth_m: float,
    camera_fov_degrees: float,
    real_hand_width_m: float,
    min_depth_m: float,
) -> np.ndarray:
    """Project normalized image-space points into metric camera space coordinates."""
    fov_rad = np.radians(float(np.clip(camera_fov_degrees, 20.0, 150.0)))
    focal_px = frame_width_px / (2.0 * np.tan(fov_rad * 0.5))
    cx = frame_width_px * 0.5
    cy = frame_height_px * 0.5

    uv = np.column_stack((
        landmarks[:, 0] * frame_width_px,
        landmarks[:, 1] * frame_height_px,
    )).astype(np.float32)

    palm_norm_width = float(
        np.linalg.norm(
            landmarks[PALM_WIDTH_PAIR[0], :2] - landmarks[PALM_WIDTH_PAIR[1], :2]
        )
    )
    meters_per_norm = real_hand_width_m / max(1e-4, palm_norm_width)

    wrist_z = float(landmarks[0, 2])
    rel_depth = (landmarks[:, 2] - wrist_z) * meters_per_norm * DEPTH_OFFSET_SCALE
    z = depth_m + rel_depth
    z = np.maximum(z, min_depth_m * 0.6)

    x = (uv[:, 0] - cx) * z / focal_px
    y = -(uv[:, 1] - cy) * z / focal_px

    return np.column_stack((x, y, z)).astype(np.float32)


class HandTracker:
    def __init__(self, config: TrackerConfig | None = None) -> None:
        self._config = config or TrackerConfig()
        model_path = self._ensure_model_file()
        self._inference_delegate_used = "cpu"
        self._hands = self._create_landmarker(model_path)
        self._last_timestamp_ms = 0

        self._landmark_filters: dict[str, AdaptiveLandmarkFilter] = {}
        self._world_filters: dict[str, AdaptiveLandmarkFilter] = {}

    @property
    def inference_delegate_used(self) -> str:
        return self._inference_delegate_used

    def _create_landmarker(self, model_path: Path) -> HandLandmarker:
        delegate_name = str(self._config.inference_delegate).lower().strip()
        if delegate_name not in {"auto", "cpu", "gpu"}:
            delegate_name = "auto"

        if delegate_name == "gpu":
            delegates = [BaseOptions.Delegate.GPU]
        elif delegate_name == "cpu":
            delegates = [BaseOptions.Delegate.CPU]
        else:
            delegates = [BaseOptions.Delegate.GPU, BaseOptions.Delegate.CPU]

        last_error: Exception | None = None
        for delegate in delegates:
            try:
                options = HandLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=str(model_path), delegate=delegate),
                    running_mode=RunningMode.VIDEO,
                    num_hands=max(1, int(self._config.max_num_hands)),
                    min_hand_detection_confidence=self._config.min_hand_detection_confidence,
                    min_hand_presence_confidence=self._config.min_hand_presence_confidence,
                    min_tracking_confidence=self._config.min_tracking_confidence,
                )
                hand_landmarker = HandLandmarker.create_from_options(options)
                self._inference_delegate_used = "gpu" if delegate == BaseOptions.Delegate.GPU else "cpu"
                return hand_landmarker
            except Exception as exc:
                last_error = exc

        if delegate_name == "gpu":
            raise RuntimeError(
                "GPU delegate requested but unavailable. "
                "Try --delegate auto or --delegate cpu."
            ) from last_error

        raise RuntimeError("Failed to initialize MediaPipe hand landmarker.") from last_error

    def process(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, HandFrameState | None]:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        results = self._hands.detect_for_video(mp_image, self._next_timestamp_ms())
        frame_h, frame_w = frame_bgr.shape[:2]

        overlay = frame_bgr.copy()

        if not results.hand_landmarks:
            self._reset_filters()
            cv2.putText(
                overlay,
                "No hands detected",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (40, 40, 240),
                2,
                cv2.LINE_AA,
            )
            return overlay, None

        hands: list[HandState] = []
        side_counter: dict[str, int] = {}
        active_keys: set[str] = set()

        for idx, hand_landmarks in enumerate(results.hand_landmarks):
            landmarks = np.asarray(
                [[lm.x, lm.y, lm.z] for lm in hand_landmarks],
                dtype=np.float32,
            )

            side, score = self._extract_handedness(results, idx, landmarks)
            side_counter[side] = side_counter.get(side, 0) + 1
            filter_key = f"{side}:{side_counter[side]}"
            active_keys.add(filter_key)

            if self._config.smoothing_enabled:
                landmarks = self._get_or_create_filter(self._landmark_filters, filter_key).apply(landmarks)

            world_landmarks: np.ndarray | None = None
            if results.hand_world_landmarks and idx < len(results.hand_world_landmarks):
                world_data = results.hand_world_landmarks[idx]
                world_landmarks = np.asarray(
                    [[lm.x, lm.y, lm.z] for lm in world_data],
                    dtype=np.float32,
                )
                if self._config.smoothing_enabled:
                    world_landmarks = self._get_or_create_filter(
                        self._world_filters,
                        filter_key,
                    ).apply(world_landmarks)

            # Always build a global camera-space hand pose from normalized image landmarks.
            # MediaPipe world landmarks are hand-local coordinates and are not directly
            # comparable across hands for inter-hand distance or camera-relative placement.
            estimated_depth_m = estimate_depth_from_hand_width(
                landmarks=landmarks,
                frame_width_px=frame_w,
                camera_fov_degrees=self._config.camera_fov_degrees,
                real_hand_width_m=self._config.default_hand_width_m,
                min_depth_m=self._config.min_depth_m,
                max_depth_m=self._config.max_depth_m,
            )
            camera_vertices = normalized_to_camera_space(
                landmarks=landmarks,
                frame_width_px=frame_w,
                frame_height_px=frame_h,
                depth_m=estimated_depth_m,
                camera_fov_degrees=self._config.camera_fov_degrees,
                real_hand_width_m=self._config.default_hand_width_m,
                min_depth_m=self._config.min_depth_m,
            )
            camera_center = np.mean(camera_vertices[PALM_CENTER_INDICES], axis=0)

            reference_landmarks = camera_vertices
            finger_flexion = compute_finger_flexion(reference_landmarks)
            bones = compute_bone_measurements(reference_landmarks)
            mesh_faces = generate_mesh_faces(landmarks)

            hands.append(
                HandState(
                    side=side,
                    handedness_score=score,
                    landmarks=landmarks,
                    world_landmarks=world_landmarks,
                    finger_flexion=finger_flexion,
                    bones=bones,
                    mesh_faces=mesh_faces,
                    camera_vertices=camera_vertices,
                    camera_center=camera_center,
                    estimated_depth_m=estimated_depth_m,
                )
            )

        self._drop_stale_filters(active_keys)
        hands.sort(key=self._hand_sort_key)

        self._draw_overlay(overlay, hands)

        return overlay, HandFrameState(hands=hands)

    def close(self) -> None:
        self._hands.close()

    def _reset_filters(self) -> None:
        self._landmark_filters.clear()
        self._world_filters.clear()

    def _drop_stale_filters(self, active_keys: set[str]) -> None:
        stale_landmark_keys = [key for key in self._landmark_filters if key not in active_keys]
        for key in stale_landmark_keys:
            del self._landmark_filters[key]

        stale_world_keys = [key for key in self._world_filters if key not in active_keys]
        for key in stale_world_keys:
            del self._world_filters[key]

    def _get_or_create_filter(
        self,
        filter_map: dict[str, AdaptiveLandmarkFilter],
        key: str,
    ) -> AdaptiveLandmarkFilter:
        if key not in filter_map:
            filter_map[key] = AdaptiveLandmarkFilter(
                slow_alpha=self._config.smoothing_slow_alpha,
                fast_alpha=self._config.smoothing_fast_alpha,
                velocity_scale=self._config.smoothing_velocity_scale,
            )
        return filter_map[key]

    @staticmethod
    def _hand_sort_key(hand: HandState) -> tuple[int, float]:
        center_x = float(hand.camera_center[0])
        return 0, center_x

    @staticmethod
    def _extract_handedness(
        results,
        index: int,
        landmarks: np.ndarray,
    ) -> tuple[str, float | None]:
        side = "Unknown"
        score: float | None = None

        handedness = getattr(results, "handedness", None)
        if handedness and index < len(handedness) and handedness[index]:
            category = handedness[index][0]
            name = (category.category_name or category.display_name or "").strip().lower()
            if name.startswith("left"):
                side = "Left"
            elif name.startswith("right"):
                side = "Right"
            if category.score is not None:
                score = float(category.score)

        if side == "Unknown":
            side = "Left" if float(np.mean(landmarks[:, 0])) < 0.5 else "Right"

        return side, score

    @staticmethod
    def _ensure_model_file() -> Path:
        model_path = Path(__file__).resolve().parent / "models" / "hand_landmarker.task"
        if model_path.exists():
            return model_path

        model_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            urllib.request.urlretrieve(DEFAULT_MODEL_URL, model_path)
        except Exception as exc:
            raise RuntimeError(
                "Failed to download hand_landmarker.task model. "
                "Please check your internet connection and retry."
            ) from exc

        return model_path

    def _next_timestamp_ms(self) -> int:
        timestamp_ms = time.monotonic_ns() // 1_000_000
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms
        return timestamp_ms

    @staticmethod
    def _palette_for_side(side: str, hand_index: int) -> tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]:
        if side == "Left":
            return (255, 192, 88), (114, 226, 255), (70, 130, 240)
        if side == "Right":
            return (96, 220, 150), (150, 235, 96), (240, 140, 70)

        if hand_index % 2 == 0:
            return (255, 192, 88), (114, 226, 255), (70, 130, 240)
        return (96, 220, 150), (150, 235, 96), (240, 140, 70)

    @classmethod
    def _draw_overlay(cls, frame: np.ndarray, hands: list[HandState]) -> None:
        height, width = frame.shape[:2]

        hand_distance_m: float | None = None
        if len(hands) >= 2:
            hand_distance_m = float(np.linalg.norm(hands[0].camera_center - hands[1].camera_center))

        for hand_index, hand in enumerate(hands):
            mesh_color, bone_color, landmark_color = cls._palette_for_side(hand.side, hand_index)

            points_px = np.column_stack(
                (hand.landmarks[:, 0] * width, hand.landmarks[:, 1] * height)
            ).astype(np.int32)

            for tri in hand.mesh_faces:
                poly = points_px[np.asarray(tri)]
                cv2.polylines(
                    frame,
                    [poly],
                    isClosed=True,
                    color=mesh_color,
                    thickness=1,
                    lineType=cv2.LINE_AA,
                )

            for start, end in BONE_CONNECTIONS:
                cv2.line(
                    frame,
                    tuple(points_px[start]),
                    tuple(points_px[end]),
                    bone_color,
                    2,
                    cv2.LINE_AA,
                )

            for point in points_px:
                cv2.circle(frame, tuple(point), 4, landmark_color, -1, cv2.LINE_AA)

            wrist_point = tuple(points_px[0])
            label = hand.side
            if hand.handedness_score is not None:
                label = f"{label} {hand.handedness_score * 100:.0f}%"
            label = f"{label} z={hand.estimated_depth_m:.2f}m"

            cv2.putText(
                frame,
                label,
                (max(10, wrist_point[0] + 10), max(20, wrist_point[1] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (240, 240, 240),
                2,
                cv2.LINE_AA,
            )

        # Metrics blocks.
        for hand_index, hand in enumerate(hands[:2]):
            if hand_index == 0:
                block_x = 20
            else:
                block_x = max(20, width - 240)

            y = 30
            title = f"{hand.side} hand"
            cv2.putText(
                frame,
                title,
                (block_x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (245, 245, 245),
                2,
                cv2.LINE_AA,
            )
            y += 22

            center = hand.camera_center
            cv2.putText(
                frame,
                f"X: {center[0]:+.2f}m",
                (block_x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (245, 245, 245),
                2,
                cv2.LINE_AA,
            )
            y += 20
            cv2.putText(
                frame,
                f"Y: {center[1]:+.2f}m",
                (block_x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (245, 245, 245),
                2,
                cv2.LINE_AA,
            )
            y += 20
            cv2.putText(
                frame,
                f"Z: {hand.estimated_depth_m:+.2f}m",
                (block_x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (245, 245, 245),
                2,
                cv2.LINE_AA,
            )
            y += 20

            for finger_name in FINGER_ORDER:
                value = float(hand.finger_flexion.get(finger_name, 0.0))
                cv2.putText(
                    frame,
                    f"{finger_name}: {value:5.1f} deg",
                    (block_x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (245, 245, 245),
                    2,
                    cv2.LINE_AA,
                )
                y += 20

        if hand_distance_m is not None:
            cv2.putText(
                frame,
                f"Hand distance: {hand_distance_m:.2f}m",
                (max(20, width // 2 - 130), 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (245, 245, 245),
                2,
                cv2.LINE_AA,
            )
