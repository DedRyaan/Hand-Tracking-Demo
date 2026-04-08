"""Optional TensorRT-backed hand tracker stub.

This module intentionally starts with a safe compatibility layer that currently
falls back to the existing MediaPipe tracker while exposing the same
HandTracker-style API. It provides a clean insertion point for future TensorRT
engine integration without breaking the app surface.
"""

from __future__ import annotations

from dataclasses import dataclass

from .tracker import HandFrameState, HandTracker, TrackerConfig


@dataclass
class TensorRTTrackerConfig:
    enabled: bool = False
    model_path: str = ""
    engine_path: str = ""
    device_id: int = 0
    fp16: bool = True


class TensorRTHandTracker:
    """Compatibility wrapper around current MediaPipe tracker.

    Today this delegates all work to `HandTracker`. A future upgrade can swap
    internals to ONNX Runtime CUDA / TensorRT while keeping the same public API.
    """

    def __init__(
        self,
        tracker_config: TrackerConfig | None = None,
        trt_config: TensorRTTrackerConfig | None = None,
    ) -> None:
        self._tracker = HandTracker(config=tracker_config or TrackerConfig())
        self._trt_config = trt_config or TensorRTTrackerConfig()
        self._backend_name = "mediapipe-cpu-gpu"

    @property
    def backend_name(self) -> str:
        return self._backend_name

    @property
    def inference_delegate_used(self) -> str:
        return self._tracker.inference_delegate_used

    def process(self, frame_bgr) -> tuple[object, HandFrameState | None]:
        return self._tracker.process(frame_bgr)

    def close(self) -> None:
        self._tracker.close()
