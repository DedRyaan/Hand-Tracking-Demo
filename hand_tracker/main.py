"""Entry point for the hand tracking GUI."""

from __future__ import annotations

import argparse
import os
import sys

# Reduce verbose native logs from MediaPipe/TFLite in console.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("GLOG_stderrthreshold", "3")
os.environ.setdefault("ABSL_LOG_LEVEL", "3")

from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication, QMessageBox

from .gui import HandTrackerWindow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time hand tracking GUI")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--fps", type=int, default=30, help="Capture update FPS (default: 30)")
    parser.add_argument(
        "--mode",
        type=str,
        default="balanced",
        choices=["balanced", "precision", "max"],
        help="Tracking mode: balanced, precision, max",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    app = QApplication(sys.argv)

    app_font = QFont("Segoe UI")
    app_font.setPointSize(10)
    app.setFont(app_font)

    try:
        window = HandTrackerWindow(camera_index=args.camera, fps=args.fps, performance_mode=args.mode)
    except RuntimeError as exc:
        QMessageBox.critical(None, "Camera Error", str(exc))
        return 1

    window.resize(1680, 940)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
