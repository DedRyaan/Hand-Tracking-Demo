"""Desktop GUI for real-time hand tracking visualization."""

from __future__ import annotations

import sys
import time

import cv2
import numpy as np
import pyqtgraph.opengl as gl
from pyqtgraph import Vector
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .constants import BONE_CONNECTIONS, FINGER_ORDER, HAND_SIDE_ORDER
from .geometry import finger_state
from .tracker import HandFrameState, HandState, HandTracker, TrackerConfig


class HandTrackerWindow(QMainWindow):
    def __init__(self, camera_index: int = 0, fps: int = 30, performance_mode: str = "balanced") -> None:
        super().__init__()
        self.setWindowTitle("Hand Tracking Studio")

        self._camera_backend_priority = self._build_backend_priority()
        self._camera_backend_active = self._camera_backend_priority[0]
        self._camera_index = camera_index
        self._performance_mode = performance_mode
        tracker_config, visual_config = self._select_mode_config(performance_mode)
        self._ui = visual_config

        self._tracker = HandTracker(config=tracker_config)
        self._camera = self._open_camera(camera_index)

        if not self._camera.isOpened():
            raise RuntimeError(f"Unable to open camera index {camera_index}.")

        self._last_frame_time = time.perf_counter()
        self._fps_smoothed = 0.0
        self._consecutive_read_failures = 0

        self._video_label = QLabel("Starting camera...")
        self._video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._video_label.setMinimumSize(920, 640)
        self._video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self._fps_label = QLabel("FPS --")
        self._fps_label.setObjectName("StatBadge")
        self._tracking_label = QLabel("Tracking idle")
        self._tracking_label.setObjectName("StatBadge")
        self._mode_label = QLabel(f"Mode {performance_mode.title()}")
        self._mode_label.setObjectName("StatBadge")
        self._hands_label = QLabel("Hands --")
        self._hands_label.setObjectName("StatBadge")
        self._distance_label = QLabel("Distance --")
        self._distance_label.setObjectName("StatBadge")

        self._camera_combo = QComboBox()
        self._camera_combo.setObjectName("SelectControl")
        self._camera_combo.setMinimumWidth(120)
        self._camera_combo.currentIndexChanged.connect(self._on_camera_combo_changed)

        self._camera_refresh_button = QPushButton("Rescan")
        self._camera_refresh_button.setObjectName("GhostButton")
        self._camera_refresh_button.clicked.connect(self._on_rescan_cameras)

        self._mode_combo = QComboBox()
        self._mode_combo.setObjectName("SelectControl")
        self._mode_combo.addItems(["balanced", "precision", "max"])
        self._mode_combo.currentTextChanged.connect(self._on_mode_combo_changed)

        self._show_landmarks = QCheckBox("Show Landmarks")
        self._show_landmarks.setChecked(self._ui["show_landmarks"])
        self._show_bones = QCheckBox("Show Bones")
        self._show_bones.setChecked(self._ui["show_bones"])
        self._show_mesh = QCheckBox("Show Mesh")
        self._show_mesh.setChecked(self._ui["show_mesh"])

        self._finger_table = self._build_finger_table()
        self._bone_table = self._build_bone_table()

        self._gl_view = gl.GLViewWidget()
        self._gl_view.setMinimumHeight(360)
        self._gl_view.setCameraPosition(distance=1.2, elevation=18, azimuth=-95)

        grid = gl.GLGridItem()
        grid.scale(0.05, 0.05, 0.05)
        grid.setDepthValue(10)
        self._gl_view.addItem(grid)

        self._hand_landmark_items: list[gl.GLScatterPlotItem] = []
        self._hand_bone_items: list[list[gl.GLLinePlotItem]] = []
        self._hand_mesh_items: list[gl.GLMeshItem] = []

        for hand_slot in range(2):
            landmark_color, bone_color, mesh_face_color, mesh_edge_color = self._palette_for_slot(hand_slot)

            landmark_item = gl.GLScatterPlotItem(
                pos=np.zeros((21, 3), dtype=np.float32),
                size=self._ui["landmark_size"],
                color=landmark_color,
                pxMode=True,
            )
            self._gl_view.addItem(landmark_item)
            self._hand_landmark_items.append(landmark_item)

            bone_items: list[gl.GLLinePlotItem] = []
            for _ in BONE_CONNECTIONS:
                bone_item = gl.GLLinePlotItem(
                    pos=np.zeros((2, 3), dtype=np.float32),
                    color=bone_color,
                    width=self._ui["bone_width"],
                    antialias=True,
                    mode="lines",
                )
                self._gl_view.addItem(bone_item)
                bone_items.append(bone_item)
            self._hand_bone_items.append(bone_items)

            mesh_item = gl.GLMeshItem(
                vertexes=np.zeros((0, 3), dtype=np.float32),
                faces=np.zeros((0, 3), dtype=np.int32),
                smooth=False,
                drawEdges=True,
                drawFaces=True,
                color=mesh_face_color,
                edgeColor=mesh_edge_color,
            )
            mesh_item.setGLOptions("translucent")
            self._gl_view.addItem(mesh_item)
            self._hand_mesh_items.append(mesh_item)

        self._layout_ui()
        self._apply_theme()

        self._populate_camera_combo(preserve_selection=False)
        self._set_mode_combo_selection(performance_mode)

        interval_ms = max(1, int(1000 / max(1, fps)))
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(interval_ms)

    @staticmethod
    def _select_mode_config(performance_mode: str) -> tuple[TrackerConfig, dict[str, object]]:
        mode = performance_mode.lower().strip()
        if mode == "max":
            tracker = TrackerConfig(
                max_num_hands=2,
                min_hand_detection_confidence=0.58,
                min_hand_presence_confidence=0.54,
                min_tracking_confidence=0.66,
                smoothing_enabled=True,
                smoothing_slow_alpha=0.5,
                smoothing_fast_alpha=0.9,
                smoothing_velocity_scale=0.024,
            )
            ui = {
                "capture_width": 960,
                "capture_height": 540,
                "capture_fps": 60,
                "landmark_size": 8,
                "bone_width": 2,
                "show_landmarks": True,
                "show_bones": True,
                "show_mesh": False,
            }
            return tracker, ui

        if mode == "precision":
            tracker = TrackerConfig(
                max_num_hands=2,
                min_hand_detection_confidence=0.66,
                min_hand_presence_confidence=0.62,
                min_tracking_confidence=0.76,
                smoothing_enabled=True,
                smoothing_slow_alpha=0.28,
                smoothing_fast_alpha=0.75,
                smoothing_velocity_scale=0.014,
            )
            ui = {
                "capture_width": 1280,
                "capture_height": 720,
                "capture_fps": 30,
                "landmark_size": 9,
                "bone_width": 2,
                "show_landmarks": True,
                "show_bones": True,
                "show_mesh": True,
            }
            return tracker, ui

        tracker = TrackerConfig(max_num_hands=2)
        ui = {
            "capture_width": 1280,
            "capture_height": 720,
            "capture_fps": 45,
            "landmark_size": 9,
            "bone_width": 2,
            "show_landmarks": True,
            "show_bones": True,
            "show_mesh": True,
        }
        return tracker, ui

    @staticmethod
    def _palette_for_slot(
        hand_slot: int,
    ) -> tuple[tuple[float, float, float, float], ...]:
        if hand_slot == 0:
            return (
                (0.18, 0.58, 0.92, 1.0),
                (0.34, 0.86, 0.72, 1.0),
                (0.14, 0.66, 0.91, 0.17),
                (0.14, 0.66, 0.91, 0.62),
            )
        return (
            (0.95, 0.58, 0.28, 1.0),
            (0.95, 0.82, 0.30, 1.0),
            (0.93, 0.62, 0.24, 0.16),
            (0.93, 0.62, 0.24, 0.60),
        )

    def _apply_theme(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow { background: #f5f7fa; }
            QWidget { color: #273449; font-family: Segoe UI, Helvetica Neue, Arial; }
            QLabel#SectionTitle { font-size: 12px; font-weight: 600; letter-spacing: 0.3px; color: #4a5568; }
            QLabel#StatBadge {
                background: #ffffff;
                border: 1px solid #dde4ee;
                border-radius: 7px;
                padding: 5px 10px;
                color: #304055;
                font-size: 11px;
                font-weight: 600;
            }
            QFrame#Card {
                background: #ffffff;
                border: 1px solid #dde4ee;
                border-radius: 12px;
            }
            QTableWidget {
                background: #ffffff;
                alternate-background-color: #f8fafc;
                border: 1px solid #e1e7f0;
                border-radius: 8px;
                gridline-color: #eef2f7;
                selection-background-color: #d8e9f8;
                selection-color: #223247;
                padding: 2px;
            }
            QHeaderView::section {
                background: #f3f7fb;
                color: #4f5f74;
                border: 0px;
                border-bottom: 1px solid #e1e7f0;
                padding: 6px 8px;
                font-weight: 600;
            }
            QCheckBox { spacing: 6px; }
            QCheckBox::indicator {
                width: 14px;
                height: 14px;
                border: 1px solid #b8c6d8;
                border-radius: 4px;
                background: #ffffff;
            }
            QCheckBox::indicator:checked {
                background: #2d9cdb;
                border: 1px solid #2d9cdb;
            }
            QComboBox#SelectControl {
                background: #ffffff;
                border: 1px solid #d6e0ec;
                border-radius: 7px;
                padding: 5px 8px;
                min-height: 16px;
            }
            QComboBox#SelectControl::drop-down {
                width: 20px;
                border: 0px;
            }
            QPushButton#GhostButton {
                background: #ffffff;
                border: 1px solid #d6e0ec;
                border-radius: 7px;
                padding: 5px 10px;
                color: #3e4d60;
                font-weight: 600;
            }
            QPushButton#GhostButton:hover {
                background: #f4f8fc;
            }
            """
        )

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
        top_bar_layout.addWidget(self._mode_label)
        top_bar_layout.addWidget(self._hands_label)
        top_bar_layout.addWidget(self._distance_label)
        top_bar_layout.addStretch(1)

        left_layout.addWidget(top_bar)
        left_layout.addWidget(self._video_label, stretch=1)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(12)

        right_card_top = QFrame()
        right_card_top.setObjectName("Card")
        right_top_layout = QVBoxLayout(right_card_top)
        right_top_layout.setContentsMargins(12, 12, 12, 12)
        right_top_layout.setSpacing(10)

        controls = QWidget()
        controls_layout = QHBoxLayout(controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.addWidget(QLabel("Camera"))
        controls_layout.addWidget(self._camera_combo)
        controls_layout.addWidget(self._camera_refresh_button)
        controls_layout.addSpacing(8)
        controls_layout.addWidget(QLabel("Mode"))
        controls_layout.addWidget(self._mode_combo)
        controls_layout.addSpacing(12)
        controls_layout.addWidget(self._show_landmarks)
        controls_layout.addWidget(self._show_bones)
        controls_layout.addWidget(self._show_mesh)
        controls_layout.addStretch(1)

        right_top_layout.addWidget(controls)
        right_top_layout.addWidget(self._gl_view, stretch=2)

        right_card_bottom = QFrame()
        right_card_bottom.setObjectName("Card")
        right_bottom_layout = QGridLayout(right_card_bottom)
        right_bottom_layout.setContentsMargins(12, 12, 12, 12)
        right_bottom_layout.setHorizontalSpacing(10)
        right_bottom_layout.setVerticalSpacing(8)

        finger_label = QLabel("Finger Movement")
        finger_label.setObjectName("SectionTitle")
        bone_label = QLabel("Bone Movement")
        bone_label.setObjectName("SectionTitle")

        right_bottom_layout.addWidget(finger_label, 0, 0)
        right_bottom_layout.addWidget(bone_label, 0, 1)
        right_bottom_layout.addWidget(self._finger_table, 1, 0)
        right_bottom_layout.addWidget(self._bone_table, 1, 1)
        right_bottom_layout.setColumnStretch(0, 1)
        right_bottom_layout.setColumnStretch(1, 1)

        right_layout.addWidget(right_card_top, stretch=2)
        right_layout.addWidget(right_card_bottom, stretch=2)

        root_layout.addWidget(left_card, stretch=3)
        root_layout.addWidget(right_panel, stretch=2)

    @staticmethod
    def _build_finger_table() -> QTableWidget:
        row_count = len(HAND_SIDE_ORDER) * len(FINGER_ORDER)
        table = QTableWidget(row_count, 4)
        table.setHorizontalHeaderLabels(["Hand", "Finger", "Flexion (deg)", "State"])
        table.verticalHeader().setVisible(False)
        table.setAlternatingRowColors(True)
        table.horizontalHeader().setStretchLastSection(True)
        table.horizontalHeader().setMinimumSectionSize(68)
        table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)

        row = 0
        for side in HAND_SIDE_ORDER:
            for name in FINGER_ORDER:
                table.setItem(row, 0, QTableWidgetItem(side))
                table.setItem(row, 1, QTableWidgetItem(name))
                table.setItem(row, 2, QTableWidgetItem("0.0"))
                table.setItem(row, 3, QTableWidgetItem("-"))
                row += 1

        table.resizeColumnsToContents()
        return table

    @staticmethod
    def _build_bone_table() -> QTableWidget:
        row_count = len(HAND_SIDE_ORDER) * len(BONE_CONNECTIONS)
        table = QTableWidget(row_count, 6)
        table.setHorizontalHeaderLabels(["Hand", "Bone", "Length", "dx", "dy", "dz"])
        table.verticalHeader().setVisible(False)
        table.setAlternatingRowColors(True)
        table.horizontalHeader().setStretchLastSection(True)
        table.horizontalHeader().setMinimumSectionSize(58)
        table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)

        row = 0
        for side in HAND_SIDE_ORDER:
            for start, end in BONE_CONNECTIONS:
                table.setItem(row, 0, QTableWidgetItem(side))
                table.setItem(row, 1, QTableWidgetItem(f"{start}->{end}"))
                table.setItem(row, 2, QTableWidgetItem("0.000"))
                table.setItem(row, 3, QTableWidgetItem("0.000"))
                table.setItem(row, 4, QTableWidgetItem("0.000"))
                table.setItem(row, 5, QTableWidgetItem("0.000"))
                row += 1

        table.resizeColumnsToContents()
        return table

    def _tick(self) -> None:
        ok, frame = self._camera.read()
        if not ok:
            self._consecutive_read_failures += 1
            if self._consecutive_read_failures >= 8:
                recovered = self._attempt_camera_recovery()
                if not recovered:
                    self._tracking_label.setText("Tracking camera unavailable")
            return

        self._consecutive_read_failures = 0

        overlay, state = self._tracker.process(frame)
        self._update_fps()
        self._update_video(overlay)

        if state is None:
            self._tracking_label.setText("Tracking no hand")
            self._hands_label.setText("Hands 0")
            self._distance_label.setText("Distance --")
            return

        self._tracking_label.setText("Tracking active")
        detected_hands = len(state.hands)
        sides = ", ".join(hand.side for hand in state.hands) if detected_hands else "none"
        self._hands_label.setText(f"Hands {detected_hands} ({sides})")
        if detected_hands >= 2:
            distance = float(np.linalg.norm(state.hands[0].camera_center - state.hands[1].camera_center))
            self._distance_label.setText(f"Distance {distance:.2f}m")
        else:
            self._distance_label.setText("Distance --")

        self._update_finger_table(state)
        self._update_bone_table(state)
        self._update_3d_view(state)

    def _update_fps(self) -> None:
        now = time.perf_counter()
        dt = max(1e-6, now - self._last_frame_time)
        instant_fps = 1.0 / dt
        if self._fps_smoothed <= 0.0:
            self._fps_smoothed = instant_fps
        else:
            self._fps_smoothed = 0.9 * self._fps_smoothed + 0.1 * instant_fps
        self._last_frame_time = now
        self._fps_label.setText(f"FPS {self._fps_smoothed:4.1f}")

    def _update_video(self, frame_bgr: np.ndarray) -> None:
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

    def _update_finger_table(self, state: HandFrameState) -> None:
        by_side: dict[str, HandState] = {hand.side: hand for hand in state.hands}

        row = 0
        for side in HAND_SIDE_ORDER:
            hand = by_side.get(side)
            for finger_name in FINGER_ORDER:
                value = float(hand.finger_flexion.get(finger_name, 0.0)) if hand is not None else 0.0
                self._finger_table.item(row, 2).setText(f"{value:5.1f}")
                self._finger_table.item(row, 3).setText(finger_state(value) if hand is not None else "-")
                row += 1

    def _update_bone_table(self, state: HandFrameState) -> None:
        by_side: dict[str, HandState] = {hand.side: hand for hand in state.hands}

        row = 0
        for side in HAND_SIDE_ORDER:
            hand = by_side.get(side)
            for bone_idx, _ in enumerate(BONE_CONNECTIONS):
                if hand is not None and bone_idx < len(hand.bones):
                    bone = hand.bones[bone_idx]
                    self._bone_table.item(row, 2).setText(f"{bone.length:.4f}")
                    self._bone_table.item(row, 3).setText(f"{bone.vector[0]:+.4f}")
                    self._bone_table.item(row, 4).setText(f"{bone.vector[1]:+.4f}")
                    self._bone_table.item(row, 5).setText(f"{bone.vector[2]:+.4f}")
                else:
                    self._bone_table.item(row, 2).setText("0.000")
                    self._bone_table.item(row, 3).setText("0.000")
                    self._bone_table.item(row, 4).setText("0.000")
                    self._bone_table.item(row, 5).setText("0.000")
                row += 1

    def _update_3d_view(self, state: HandFrameState) -> None:
        show_landmarks = self._show_landmarks.isChecked()
        show_bones = self._show_bones.isChecked()
        show_mesh = self._show_mesh.isChecked()

        if state.hands:
            centers = np.asarray([hand.camera_center for hand in state.hands], dtype=np.float32)
            scene_center = np.mean(centers, axis=0)
            spread = float(np.max(np.linalg.norm(centers - scene_center, axis=1))) if len(centers) > 1 else 0.12
            nearest_z = float(np.min(centers[:, 2]))
            target_distance = max(0.35, min(2.0, nearest_z + 0.45 + spread * 1.5))
        else:
            scene_center = np.zeros(3, dtype=np.float32)
            target_distance = 1.0

        self._gl_view.opts["center"] = Vector(
            float(scene_center[0]),
            float(scene_center[1]),
            float(scene_center[2]),
        )
        self._gl_view.opts["distance"] = float(target_distance)

        for slot, hand in enumerate(state.hands[:2]):
            vertices = hand.camera_vertices

            landmark_item = self._hand_landmark_items[slot]
            landmark_color, _, _, _ = self._palette_for_slot(slot)
            landmark_item.setVisible(show_landmarks)
            landmark_item.setData(
                pos=vertices,
                size=self._ui["landmark_size"],
                color=landmark_color,
            )

            for line_item, (start, end) in zip(self._hand_bone_items[slot], BONE_CONNECTIONS, strict=True):
                line_item.setVisible(show_bones)
                line_item.setData(pos=np.vstack((vertices[start], vertices[end])))

            mesh_item = self._hand_mesh_items[slot]
            mesh_item.setVisible(show_mesh)
            if show_mesh and len(hand.mesh_faces) > 0:
                mesh_item.setMeshData(vertexes=vertices, faces=hand.mesh_faces)

        for slot in range(len(state.hands), 2):
            self._hand_landmark_items[slot].setVisible(False)
            for line_item in self._hand_bone_items[slot]:
                line_item.setVisible(False)
            self._hand_mesh_items[slot].setVisible(False)

    @staticmethod
    def _build_backend_priority() -> list[int]:
        if sys.platform.startswith("win"):
            return [cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY]
        return [cv2.CAP_ANY]

    @staticmethod
    def _backend_name(backend: int) -> str:
        names = {
            cv2.CAP_ANY: "AUTO",
            cv2.CAP_DSHOW: "DSHOW",
            cv2.CAP_MSMF: "MSMF",
        }
        return names.get(backend, str(backend))

    def _try_open_camera_with_backends(self, camera_index: int) -> tuple[cv2.VideoCapture | None, int | None]:
        for backend in self._camera_backend_priority:
            camera = cv2.VideoCapture(camera_index, backend)
            if not camera.isOpened():
                camera.release()
                continue
            self._apply_capture_profile(camera)
            ok, _ = camera.read()
            if ok:
                return camera, backend
            camera.release()
        return None, None

    def _open_camera(self, camera_index: int) -> cv2.VideoCapture:
        camera, backend = self._try_open_camera_with_backends(camera_index)
        if camera is not None and backend is not None:
            self._camera_backend_active = backend
            return camera

        fallback = cv2.VideoCapture(camera_index)
        if fallback.isOpened():
            self._apply_capture_profile(fallback)
            self._camera_backend_active = cv2.CAP_ANY
        return fallback

    def _apply_capture_profile(self, camera: cv2.VideoCapture) -> None:
        width = int(self._ui["capture_width"])
        height = int(self._ui["capture_height"])
        fps = int(self._ui["capture_fps"])

        profiles = [
            (width, height, fps),
            (960, 540, min(30, fps)),
            (640, 480, min(30, fps)),
        ]

        for w, h, f in profiles:
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, float(w))
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, float(h))
            camera.set(cv2.CAP_PROP_FPS, float(max(15, f)))
            ok, _ = camera.read()
            if ok:
                break

        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def _list_available_camera_indices(self, max_index: int = 6) -> list[int]:
        available: list[int] = []
        for idx in range(max_index + 1):
            cap, _ = self._try_open_camera_with_backends(idx)
            if cap is not None:
                available.append(idx)
                cap.release()
        return available

    def _populate_camera_combo(self, preserve_selection: bool = True) -> None:
        current_value = self._camera_index if preserve_selection else None
        cameras = self._list_available_camera_indices()
        if self._camera_index not in cameras:
            cameras.append(self._camera_index)
            cameras.sort()

        self._camera_combo.blockSignals(True)
        self._camera_combo.clear()
        for idx in cameras:
            self._camera_combo.addItem(f"Camera {idx}", idx)

        target_index = self._camera_index if current_value is None else current_value
        combo_index = self._camera_combo.findData(target_index)
        if combo_index >= 0:
            self._camera_combo.setCurrentIndex(combo_index)
        self._camera_combo.blockSignals(False)

    def _set_mode_combo_selection(self, mode: str) -> None:
        self._mode_combo.blockSignals(True)
        idx = self._mode_combo.findText(mode.lower())
        if idx >= 0:
            self._mode_combo.setCurrentIndex(idx)
        self._mode_combo.blockSignals(False)

    def _on_rescan_cameras(self) -> None:
        self._populate_camera_combo(preserve_selection=True)

    def _on_camera_combo_changed(self, index: int) -> None:
        if index < 0:
            return
        data = self._camera_combo.itemData(index)
        if data is None:
            return
        camera_index = int(data)
        if camera_index == self._camera_index:
            return
        self._switch_camera(camera_index)

    def _switch_camera(self, camera_index: int) -> None:
        new_camera = self._open_camera(camera_index)
        if not new_camera.isOpened():
            self._tracking_label.setText(f"Tracking camera {camera_index} unavailable")
            new_camera.release()
            self._populate_camera_combo(preserve_selection=True)
            return

        self._camera.release()
        self._camera = new_camera
        self._camera_index = camera_index
        self._consecutive_read_failures = 0
        self._fps_smoothed = 0.0
        self._last_frame_time = time.perf_counter()
        self._distance_label.setText("Distance --")
        self._distance_label.setText("Distance --")
        backend = self._backend_name(self._camera_backend_active)
        self._tracking_label.setText(f"Tracking camera {camera_index} active ({backend})")
        self._populate_camera_combo(preserve_selection=True)

    def _on_mode_combo_changed(self, mode: str) -> None:
        target = mode.lower().strip()
        if target == self._performance_mode:
            return
        self._apply_mode(target)

    def _apply_mode(self, performance_mode: str) -> None:
        tracker_config, visual_config = self._select_mode_config(performance_mode)

        new_camera = self._open_camera(self._camera_index)
        if not new_camera.isOpened():
            self._tracking_label.setText(f"Tracking camera {self._camera_index} unavailable")
            new_camera.release()
            self._set_mode_combo_selection(self._performance_mode)
            return

        new_tracker = HandTracker(config=tracker_config)

        self._camera.release()
        self._tracker.close()
        self._camera = new_camera
        self._tracker = new_tracker

        self._performance_mode = performance_mode
        self._ui = visual_config

        self._mode_label.setText(f"Mode {performance_mode.title()}")

        self._show_landmarks.setChecked(bool(self._ui["show_landmarks"]))
        self._show_bones.setChecked(bool(self._ui["show_bones"]))
        self._show_mesh.setChecked(bool(self._ui["show_mesh"]))

        self._consecutive_read_failures = 0
        self._fps_smoothed = 0.0
        self._last_frame_time = time.perf_counter()

    def _attempt_camera_recovery(self) -> bool:
        new_camera = self._open_camera(self._camera_index)
        if not new_camera.isOpened():
            new_camera.release()
            return False

        self._camera.release()
        self._camera = new_camera
        self._consecutive_read_failures = 0
        self._fps_smoothed = 0.0
        self._last_frame_time = time.perf_counter()
        self._distance_label.setText("Distance --")

        backend = self._backend_name(self._camera_backend_active)
        self._tracking_label.setText(f"Tracking recovered ({backend})")
        return True

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._timer.stop()
        self._camera.release()
        self._tracker.close()
        super().closeEvent(event)
