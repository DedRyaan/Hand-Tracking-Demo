"""Desktop air-mouse app powered by the hand_tracker pipeline."""

from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QFont, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from hand_tracker.tracker import HandTracker, TrackerConfig

from .controller import AirMouseConfig, AirMouseController


os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("GLOG_stderrthreshold", "3")
os.environ.setdefault("ABSL_LOG_LEVEL", "3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Air Mouse hand-tracking controller")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--fps", type=int, default=45, help="Capture update FPS (default: 45)")
    parser.add_argument(
        "--mode",
        type=str,
        default="balanced",
        choices=["balanced", "precision", "max"],
        help="Tracking mode: balanced, precision, max",
    )
    parser.add_argument(
        "--control-hand",
        type=str,
        default="right",
        choices=["right", "left", "auto"],
        help="Preferred hand for pointer control",
    )
    parser.add_argument(
        "--no-mirror",
        action="store_true",
        help="Disable horizontal mirroring for cursor movement",
    )
    parser.add_argument(
        "--delegate",
        type=str,
        default="auto",
        choices=["auto", "gpu", "cpu"],
        help="MediaPipe inference delegate: auto, gpu, cpu",
    )
    return parser.parse_args()


class AirMouseWindow(QMainWindow):
    def __init__(
        self,
        camera_index: int = 0,
        fps: int = 45,
        performance_mode: str = "balanced",
        control_hand: str = "right",
        mirror_x: bool = True,
        delegate: str = "auto",
    ) -> None:
        super().__init__()
        self.setWindowTitle("Air Mouse Studio")

        self._camera_index = camera_index
        self._fps_target = max(1, int(fps))

        self._performance_mode = performance_mode.lower().strip()
        tracker_config, capture_profile = self._select_mode_config(self._performance_mode)
        tracker_config.inference_delegate = delegate.lower().strip()
        self._capture_profile = capture_profile

        self._tracker = HandTracker(config=tracker_config)
        self._controller = AirMouseController(
            AirMouseConfig(
                control_hand_preference=control_hand,
                mirror_x=mirror_x,
                swap_handedness_labels=False,
                strict_hand_selection=True,
            )
        )

        self._camera = cv2.VideoCapture(camera_index)
        if not self._camera.isOpened():
            raise RuntimeError(f"Unable to open camera index {camera_index}.")
        self._apply_capture_profile()

        screen_w, screen_h = self._controller.screen_size

        self._last_frame_time = time.perf_counter()
        self._fps_smoothed = 0.0

        self._video_label = QLabel("Starting camera...")
        self._video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._video_label.setMinimumSize(900, 620)
        self._video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self._fps_label = QLabel("FPS --")
        self._tracking_label = QLabel("Tracking idle")
        self._hands_label = QLabel("Hands --")
        self._control_label = QLabel("Control hand --")
        self._action_label = QLabel("Last action Idle")
        self._delegate_label = QLabel(f"Delegate {self._tracker.inference_delegate_used.upper()}")

        for badge in (
            self._fps_label,
            self._tracking_label,
            self._hands_label,
            self._control_label,
            self._action_label,
            self._delegate_label,
        ):
            badge.setObjectName("Badge")

        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["balanced", "precision", "max"])
        self._mode_combo.currentTextChanged.connect(self._on_mode_changed)
        self._set_mode_combo_selection(self._performance_mode)

        self._control_combo = QComboBox()
        self._control_combo.addItems(["right", "left", "auto"])
        self._control_combo.currentTextChanged.connect(self._on_control_hand_changed)
        self._set_control_combo_selection(control_hand)

        shortcuts = "\n".join(f"- {line}" for line in AirMouseController.shortcut_reference())
        self._shortcuts_label = QLabel(
            "Mouse controls\n"
            "- Index + thumb pinch: left click\n"
            "- Middle + thumb pinch: right click\n"
            "- Both pinches together: double click\n"
            "- Fist hold: drag while moving\n\n"
            f"Screen mapping\n- {screen_w} x {screen_h}\n\n"
            "Pointer zone\n"
            "- Center region controls full screen (no full-FOV reach needed)\n"
            "- You can hold your hand mostly around the middle of camera view\n"
            "- Boosted mapping enabled (small movement reaches full screen)\n\n"
            "Click stability\n"
            "- Pointer micro-jitter is damped during pinch/fist click intent\n"
            "- Pinch click uses a tiny hold confirm for reliable activation\n\n"
            "Shortcuts\n"
            "- Open palm (front) is free movement only\n"
            "- Show desktop requires German three sign + short hold\n"
            f"{shortcuts}"
        )
        self._shortcuts_label.setObjectName("HelpText")
        self._shortcuts_label.setWordWrap(True)

        self._layout_ui()
        self._apply_theme()

        interval_ms = max(1, int(1000 / self._fps_target))
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(interval_ms)

    @staticmethod
    def _select_mode_config(performance_mode: str) -> tuple[TrackerConfig, tuple[int, int, int]]:
        mode = performance_mode.lower().strip()
        if mode == "max":
            tracker = TrackerConfig(
                max_num_hands=2,
                min_hand_detection_confidence=0.58,
                min_hand_presence_confidence=0.54,
                min_tracking_confidence=0.66,
                smoothing_enabled=True,
                smoothing_slow_alpha=0.48,
                smoothing_fast_alpha=0.9,
                smoothing_velocity_scale=0.026,
            )
            return tracker, (960, 540, 60)

        if mode == "precision":
            tracker = TrackerConfig(
                max_num_hands=2,
                min_hand_detection_confidence=0.66,
                min_hand_presence_confidence=0.62,
                min_tracking_confidence=0.76,
                smoothing_enabled=True,
                smoothing_slow_alpha=0.28,
                smoothing_fast_alpha=0.74,
                smoothing_velocity_scale=0.014,
            )
            return tracker, (1280, 720, 30)

        return TrackerConfig(max_num_hands=2), (1280, 720, 45)

    def _layout_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)

        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(14, 14, 14, 14)
        root_layout.setSpacing(14)

        left_card = QFrame()
        left_card.setObjectName("Card")
        left_layout = QVBoxLayout(left_card)
        left_layout.setContentsMargins(12, 12, 12, 12)
        left_layout.setSpacing(8)

        top_bar = QWidget()
        top_bar_layout = QHBoxLayout(top_bar)
        top_bar_layout.setContentsMargins(0, 0, 0, 0)
        top_bar_layout.setSpacing(8)
        top_bar_layout.addWidget(self._fps_label)
        top_bar_layout.addWidget(self._tracking_label)
        top_bar_layout.addWidget(self._hands_label)
        top_bar_layout.addWidget(self._control_label)
        top_bar_layout.addWidget(self._delegate_label)
        top_bar_layout.addWidget(self._action_label, stretch=1)

        left_layout.addWidget(top_bar)
        left_layout.addWidget(self._video_label, stretch=1)

        right_card = QFrame()
        right_card.setObjectName("Card")
        right_layout = QVBoxLayout(right_card)
        right_layout.setContentsMargins(12, 12, 12, 12)
        right_layout.setSpacing(10)

        controls = QWidget()
        controls_layout = QHBoxLayout(controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(8)
        controls_layout.addWidget(QLabel("Mode"))
        controls_layout.addWidget(self._mode_combo)
        controls_layout.addSpacing(10)
        controls_layout.addWidget(QLabel("Pointer"))
        controls_layout.addWidget(self._control_combo)
        controls_layout.addStretch(1)

        right_layout.addWidget(controls)
        right_layout.addWidget(self._shortcuts_label, stretch=1)

        root_layout.addWidget(left_card, stretch=3)
        root_layout.addWidget(right_card, stretch=2)

    def _apply_theme(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow { background: #eef3f8; }
            QWidget { color: #243344; font-family: Segoe UI, Helvetica Neue, Arial; }
            QFrame#Card {
                background: #ffffff;
                border: 1px solid #d7e1ed;
                border-radius: 12px;
            }
            QLabel#Badge {
                background: #f8fbff;
                border: 1px solid #d6e0ec;
                border-radius: 7px;
                padding: 5px 10px;
                font-size: 11px;
                font-weight: 600;
                color: #2f4257;
            }
            QLabel#HelpText {
                background: #f8fbff;
                border: 1px solid #d6e0ec;
                border-radius: 10px;
                padding: 10px;
                color: #31475e;
                font-size: 12px;
                line-height: 1.4;
            }
            QComboBox {
                background: #ffffff;
                border: 1px solid #ced9e6;
                border-radius: 7px;
                padding: 5px 8px;
                min-width: 92px;
            }
            QComboBox::drop-down { width: 20px; border: 0px; }
            """
        )

    def _set_mode_combo_selection(self, mode: str) -> None:
        self._mode_combo.blockSignals(True)
        idx = self._mode_combo.findText(mode.lower())
        if idx >= 0:
            self._mode_combo.setCurrentIndex(idx)
        self._mode_combo.blockSignals(False)

    def _set_control_combo_selection(self, control_hand: str) -> None:
        self._control_combo.blockSignals(True)
        idx = self._control_combo.findText(control_hand.lower())
        if idx >= 0:
            self._control_combo.setCurrentIndex(idx)
        self._control_combo.blockSignals(False)

    def _apply_capture_profile(self) -> None:
        width, height, fps = self._capture_profile
        self._camera.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
        self._camera.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
        self._camera.set(cv2.CAP_PROP_FPS, float(max(15, fps)))
        self._camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def _tick(self) -> None:
        ok, frame = self._camera.read()
        if not ok:
            self._tracking_label.setText("Tracking camera unavailable")
            self._controller.on_no_hands()
            return

        overlay, state = self._tracker.process(frame)
        self._update_fps()

        if state is None:
            self._controller.on_no_hands()
            self._tracking_label.setText("Tracking no hand")
            self._hands_label.setText("Hands 0")
            self._control_label.setText("Control hand --")
            self._action_label.setText(f"Last action {self._controller.latest_action()}")
            self._update_video(overlay)
            return

        actions: list[str] = []
        try:
            actions = self._controller.update(state)
        except Exception as exc:
            self._tracking_label.setText(f"Input error: {exc}")

        hands = len(state.hands)
        sides = ", ".join(hand.side for hand in state.hands)
        self._tracking_label.setText("Tracking active")
        self._hands_label.setText(f"Hands {hands} ({sides})")
        self._control_label.setText(f"Control hand {self._controller.last_primary_side}")

        latest = actions[-1] if actions else self._controller.latest_action()
        suffix = " (dragging)" if self._controller.drag_active else ""
        self._action_label.setText(f"Last action {latest}{suffix}")

        if actions:
            cv2.putText(
                overlay,
                f"Action: {actions[-1]}",
                (20, max(36, overlay.shape[0] - 24)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (245, 245, 245),
                2,
                cv2.LINE_AA,
            )

        self._update_video(overlay)

    def _update_fps(self) -> None:
        now = time.perf_counter()
        dt = max(1e-6, now - self._last_frame_time)
        instant_fps = 1.0 / dt
        if self._fps_smoothed <= 0.0:
            self._fps_smoothed = instant_fps
        else:
            self._fps_smoothed = 0.88 * self._fps_smoothed + 0.12 * instant_fps
        self._last_frame_time = now
        self._fps_label.setText(f"FPS {self._fps_smoothed:4.1f}")

    def _update_video(self, frame_bgr) -> None:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, channels = frame_rgb.shape
        bytes_per_line = channels * w
        image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(image)
        self._video_label.setPixmap(
            pixmap.scaled(
                self._video_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def _on_mode_changed(self, mode: str) -> None:
        target_mode = mode.lower().strip()
        if target_mode == self._performance_mode:
            return

        tracker_config, capture_profile = self._select_mode_config(target_mode)
        tracker_config.inference_delegate = self._tracker.inference_delegate_used
        new_tracker = HandTracker(config=tracker_config)

        self._tracker.close()
        self._tracker = new_tracker

        self._performance_mode = target_mode
        self._capture_profile = capture_profile
        self._delegate_label.setText(f"Delegate {self._tracker.inference_delegate_used.upper()}")
        self._apply_capture_profile()

    def _on_control_hand_changed(self, hand_name: str) -> None:
        self._controller.set_control_hand_preference(hand_name)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._timer.stop()
        self._camera.release()
        self._tracker.close()
        self._controller.close()
        super().closeEvent(event)


def main() -> int:
    args = parse_args()
    app = QApplication(sys.argv)

    app_font = QFont("Segoe UI")
    app_font.setPointSize(10)
    app.setFont(app_font)

    try:
        window = AirMouseWindow(
            camera_index=args.camera,
            fps=args.fps,
            performance_mode=args.mode,
            control_hand=args.control_hand,
            mirror_x=not args.no_mirror,
            delegate=args.delegate,
        )
    except RuntimeError as exc:
        QMessageBox.critical(None, "Air Mouse Error", str(exc))
        return 1

    window.resize(1620, 920)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
