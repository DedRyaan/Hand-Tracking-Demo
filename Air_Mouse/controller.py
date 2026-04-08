"""Mouse and shortcut emulation driven by hand gestures."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import sys
import time
from typing import Callable

import numpy as np
from pynput.keyboard import Controller as KeyboardController
from pynput.keyboard import Key
from pynput.mouse import Button
from pynput.mouse import Controller as MouseController

from hand_tracker.tracker import HandFrameState, HandState

from .gestures import HandGestureState, classify_hand_gesture, index_tip_xy, palm_center_xy


@dataclass
class AirMouseConfig:
    control_hand_preference: str = "right"
    interaction_mode: str = "pinch"
    swap_handedness_labels: bool = False
    mirror_x: bool = True
    cursor_smoothing_alpha: float = 0.22
    cursor_deadzone_px: float = 4.5
    cursor_motion_fast_px: float = 120.0
    pointer_history_size: int = 5
    active_region_padding: float = 0.42
    pinch_freeze_deadzone_px: float = 11.0
    pinch_press_left_ratio: float = 0.56
    pinch_release_left_ratio: float = 0.72
    pinch_press_right_ratio: float = 0.58
    pinch_release_right_ratio: float = 0.74
    pinch_min_active_sec: float = 0.015
    click_cooldown_sec: float = 0.16
    click_cursor_lock_sec: float = 0.14
    dwell_click_sec: float = 0.55
    dwell_radius_px: float = 16.0
    dwell_rearm_move_px: float = 30.0
    shortcut_cooldown_sec: float = 1.0
    desktop_hold_sec: float = 0.3
    strict_hand_selection: bool = True


class AirMouseController:
    def __init__(self, config: AirMouseConfig | None = None) -> None:
        self._config = config or AirMouseConfig()
        self._mouse = MouseController()
        self._keyboard = KeyboardController()

        self._screen_width, self._screen_height = self._detect_screen_size()
        self._filtered_cursor: np.ndarray | None = None
        self._cursor_history: deque[np.ndarray] = deque(
            maxlen=max(1, int(self._config.pointer_history_size))
        )
        self._cursor_lock_until = 0.0

        self._drag_active = False

        self._left_pinch_prev = False
        self._middle_pinch_prev = False
        self._dual_pinch_prev = False
        self._left_pinch_armed = False
        self._right_pinch_armed = False
        self._left_pinch_down_since = 0.0
        self._right_pinch_down_since = 0.0
        self._dual_pinch_active = False

        self._dwell_anchor: np.ndarray | None = None
        self._dwell_start = 0.0
        self._dwell_fired = False

        self._gesture_prev: dict[str, bool] = {}
        self._gesture_last_fire: dict[str, float] = {}
        self._desktop_hold_start_by_side: dict[str, float] = {}

        self._last_primary_side = "--"
        self._last_action = "Idle"
        self._last_action_time = time.monotonic()

    def _normalize_side(self, side: str) -> str:
        if not self._config.swap_handedness_labels:
            return side
        if side == "Left":
            return "Right"
        if side == "Right":
            return "Left"
        return side

    def _desktop_gesture_ready(self, hand: HandState, gesture: HandGestureState) -> bool:
        if not gesture.german_three_sign:
            self._desktop_hold_start_by_side.pop(hand.side, None)
            return False

        now = time.monotonic()
        started = self._desktop_hold_start_by_side.get(hand.side)
        if started is None:
            self._desktop_hold_start_by_side[hand.side] = now
            return False

        return (now - started) >= float(self._config.desktop_hold_sec)

    @property
    def screen_size(self) -> tuple[int, int]:
        return self._screen_width, self._screen_height

    @property
    def config(self) -> AirMouseConfig:
        return self._config

    def set_control_hand_preference(self, control_hand: str) -> None:
        name = control_hand.lower().strip()
        if name not in {"left", "right", "auto"}:
            return
        self._config.control_hand_preference = name
        self._last_primary_side = "--"
        self.on_no_hands()

    def set_interaction_mode(self, interaction_mode: str) -> None:
        mode = interaction_mode.lower().strip()
        if mode not in {"pinch", "two_hand", "dwell"}:
            return
        self._config.interaction_mode = mode
        self.on_no_hands()

    @property
    def last_primary_side(self) -> str:
        return self._last_primary_side

    @property
    def drag_active(self) -> bool:
        return self._drag_active

    @property
    def interaction_mode(self) -> str:
        return self._config.interaction_mode

    def latest_action(self) -> str:
        if time.monotonic() - self._last_action_time > 2.8:
            return "Idle"
        return self._last_action

    def update(self, frame_state: HandFrameState) -> list[str]:
        actions: list[str] = []

        primary_hand, secondary_hand = self._select_hands(frame_state.hands)
        if primary_hand is None:
            self._last_primary_side = "--"
            self.on_no_hands()
            return actions

        self._last_primary_side = self._normalize_side(primary_hand.side)
        primary_gesture = classify_hand_gesture(primary_hand)
        secondary_gesture = classify_hand_gesture(secondary_hand) if secondary_hand is not None else None

        mode = self._config.interaction_mode
        if mode == "two_hand":
            self._handle_clicks_two_hand(secondary_hand, secondary_gesture, actions)
        elif mode == "dwell":
            self._handle_clicks_dwell(primary_gesture, actions)
        else:
            self._handle_clicks_pinch(primary_gesture, actions, freeze_cursor=True)

        self._move_cursor(primary_hand, primary_gesture)

        if mode == "two_hand" and secondary_hand is not None and secondary_gesture is not None:
            shortcut_hand = secondary_hand
            shortcut_source = secondary_gesture
        else:
            shortcut_hand = secondary_hand if secondary_hand is not None else primary_hand
            shortcut_source = secondary_gesture if secondary_gesture is not None else primary_gesture

        self._handle_shortcuts(shortcut_hand, shortcut_source, actions)

        return actions

    def on_no_hands(self) -> None:
        self._left_pinch_prev = False
        self._middle_pinch_prev = False
        self._dual_pinch_prev = False
        self._left_pinch_armed = False
        self._right_pinch_armed = False
        self._left_pinch_down_since = 0.0
        self._right_pinch_down_since = 0.0
        self._dual_pinch_active = False
        self._dwell_anchor = None
        self._dwell_start = 0.0
        self._dwell_fired = False
        self._cursor_lock_until = 0.0
        self._filtered_cursor = None
        self._cursor_history.clear()
        self._desktop_hold_start_by_side.clear()

        for token in list(self._gesture_prev.keys()):
            self._gesture_prev[token] = False

        if self._drag_active:
            self._mouse.release(Button.left)
            self._drag_active = False
            self._remember_action("Drag end")

    def close(self) -> None:
        if self._drag_active:
            self._mouse.release(Button.left)
            self._drag_active = False

    @staticmethod
    def shortcut_reference() -> list[str]:
        return [
            "Peace sign  -> Back (Alt+Left)",
            "Three fingers -> Forward (Alt+Right)",
            "Thumbs up -> Play/Pause",
            "German 3 sign (thumb+index+middle) -> Show desktop",
        ]

    @staticmethod
    def _detect_screen_size() -> tuple[int, int]:
        try:
            import tkinter

            root = tkinter.Tk()
            root.withdraw()
            root.update_idletasks()
            width = int(root.winfo_screenwidth())
            height = int(root.winfo_screenheight())
            root.destroy()
            return max(640, width), max(480, height)
        except Exception:
            return 1920, 1080

    def _select_hands(self, hands: list[HandState]) -> tuple[HandState | None, HandState | None]:
        if not hands:
            return None, None

        preference = self._config.control_hand_preference.lower().strip()

        def side_matches(hand: HandState, target_side: str) -> bool:
            return self._normalize_side(hand.side) == target_side

        if preference == "left":
            primary = next((hand for hand in hands if side_matches(hand, "Left")), None)
        elif preference == "auto":
            primary = next((hand for hand in hands if side_matches(hand, "Right")), None)
            if primary is None:
                primary = next((hand for hand in hands if side_matches(hand, "Left")), None)
        else:
            primary = next((hand for hand in hands if side_matches(hand, "Right")), None)

        if primary is None and self._config.strict_hand_selection and preference in {"left", "right"}:
            return None, None

        if primary is None:
            primary = hands[0]

        secondary = next((hand for hand in hands if hand is not primary), None)
        return primary, secondary

    def _move_cursor(self, primary_hand: HandState, gesture: HandGestureState) -> None:
        now = time.monotonic()
        if now < self._cursor_lock_until:
            return

        raw_xy = index_tip_xy(primary_hand)
        palm_xy = palm_center_xy(primary_hand)
        x = float(np.clip(raw_xy[0], 0.0, 1.0))
        y = float(np.clip(raw_xy[1], 0.0, 1.0))
        palm_x = float(np.clip(palm_xy[0], 0.0, 1.0))
        palm_y = float(np.clip(palm_xy[1], 0.0, 1.0))

        if self._config.mirror_x:
            x = 1.0 - x
            palm_x = 1.0 - palm_x

        pad = float(np.clip(self._config.active_region_padding, 0.0, 0.45))
        active_span = max(1e-4, 1.0 - 2.0 * pad)
        x = float(np.clip((x - pad) / active_span, 0.0, 1.0))
        y = float(np.clip((y - pad) / active_span, 0.0, 1.0))
        palm_x = float(np.clip((palm_x - pad) / active_span, 0.0, 1.0))
        palm_y = float(np.clip((palm_y - pad) / active_span, 0.0, 1.0))

        index_target = np.asarray(
            [x * (self._screen_width - 1), y * (self._screen_height - 1)], dtype=np.float32
        )
        palm_target = np.asarray(
            [palm_x * (self._screen_width - 1), palm_y * (self._screen_height - 1)], dtype=np.float32
        )

        if self._drag_active or gesture.fist:
            index_weight = 0.30
        elif gesture.index_thumb_pinch or gesture.middle_thumb_pinch:
            index_weight = 0.46
        else:
            index_weight = 0.78

        target = index_weight * index_target + (1.0 - index_weight) * palm_target
        self._cursor_history.append(target)
        if len(self._cursor_history) >= 2:
            history = np.asarray(self._cursor_history, dtype=np.float32)
            target = np.mean(history, axis=0)

        if self._filtered_cursor is None:
            self._filtered_cursor = target
        else:
            motion = float(np.linalg.norm(target - self._filtered_cursor))
            base_alpha = float(np.clip(self._config.cursor_smoothing_alpha, 0.01, 0.95))
            fast_px = max(10.0, float(self._config.cursor_motion_fast_px))
            motion_boost = float(np.clip(motion / fast_px, 0.0, 1.0))
            alpha = base_alpha + (0.78 - base_alpha) * motion_boost

            if motion < float(self._config.cursor_deadzone_px):
                alpha = min(alpha, 0.08)

            click_intent = False
            if self._config.interaction_mode in {"pinch", "two_hand"}:
                click_intent = (
                    self._left_pinch_armed
                    or self._right_pinch_armed
                    or self._dual_pinch_active
                    or gesture.fist
                )
            elif self._config.interaction_mode == "dwell":
                click_intent = self._dwell_start > 0.0

            if click_intent and motion < float(self._config.pinch_freeze_deadzone_px):
                alpha = min(alpha, 0.03)

            self._filtered_cursor = (1.0 - alpha) * self._filtered_cursor + alpha * target

        cursor = np.clip(
            self._filtered_cursor,
            [0.0, 0.0],
            [float(self._screen_width - 1), float(self._screen_height - 1)],
        )
        self._mouse.position = int(cursor[0]), int(cursor[1])

    def _reset_pinch_state(self) -> None:
        self._left_pinch_prev = False
        self._middle_pinch_prev = False
        self._dual_pinch_prev = False
        self._left_pinch_armed = False
        self._right_pinch_armed = False
        self._left_pinch_down_since = 0.0
        self._right_pinch_down_since = 0.0
        self._dual_pinch_active = False

    def _handle_clicks_pinch(
        self,
        gesture: HandGestureState,
        actions: list[str],
        freeze_cursor: bool,
    ) -> None:
        now = time.monotonic()

        left_press = gesture.index_thumb_ratio <= float(self._config.pinch_press_left_ratio)
        left_release = gesture.index_thumb_ratio >= float(self._config.pinch_release_left_ratio)
        right_press = gesture.middle_thumb_ratio <= float(self._config.pinch_press_right_ratio)
        right_release = gesture.middle_thumb_ratio >= float(self._config.pinch_release_right_ratio)

        left_release_event = False
        right_release_event = False

        if left_press:
            if self._left_pinch_down_since <= 0.0:
                self._left_pinch_down_since = now
            if (now - self._left_pinch_down_since) >= float(self._config.pinch_min_active_sec):
                self._left_pinch_armed = True
        elif left_release:
            left_release_event = self._left_pinch_armed
            self._left_pinch_down_since = 0.0
            self._left_pinch_armed = False

        if right_press:
            if self._right_pinch_down_since <= 0.0:
                self._right_pinch_down_since = now
            if (now - self._right_pinch_down_since) >= float(self._config.pinch_min_active_sec):
                self._right_pinch_armed = True
        elif right_release:
            right_release_event = self._right_pinch_armed
            self._right_pinch_down_since = 0.0
            self._right_pinch_armed = False

        dual_pinch_live = self._left_pinch_armed and self._right_pinch_armed
        if dual_pinch_live:
            self._dual_pinch_active = True

        pinch_contact = (
            left_press
            or right_press
            or self._left_pinch_armed
            or self._right_pinch_armed
            or self._dual_pinch_active
        )

        if freeze_cursor and pinch_contact:
            self._cursor_lock_until = max(
                self._cursor_lock_until,
                now + float(self._config.click_cursor_lock_sec),
            )

        if (
            self._dual_pinch_active
            and left_release_event
            and right_release_event
            and not self._drag_active
            and self._check_cooldown("double_click", now, self._config.click_cooldown_sec)
        ):
            self._mouse.click(Button.left, 2)
            self._note_action(actions, "Double click")
            self._dual_pinch_active = False

        if gesture.fist and not self._drag_active:
            self._mouse.press(Button.left)
            self._drag_active = True
            self._cursor_lock_until = max(
                self._cursor_lock_until,
                now + float(self._config.click_cursor_lock_sec),
            )
            self._note_action(actions, "Drag start")
        elif not gesture.fist and self._drag_active:
            self._mouse.release(Button.left)
            self._drag_active = False
            self._cursor_lock_until = max(
                self._cursor_lock_until,
                now + float(self._config.click_cursor_lock_sec),
            )
            self._note_action(actions, "Drag end")

        if (
            left_release_event
            and not self._drag_active
            and not self._dual_pinch_active
            and self._check_cooldown("left_click", now, self._config.click_cooldown_sec)
        ):
            self._mouse.click(Button.left, 1)
            self._note_action(actions, "Left click")

        if (
            right_release_event
            and not self._dual_pinch_active
            and self._check_cooldown("right_click", now, self._config.click_cooldown_sec)
        ):
            self._mouse.click(Button.right, 1)
            self._note_action(actions, "Right click")

        if left_release_event and right_release_event:
            self._dual_pinch_active = False

        self._left_pinch_prev = self._left_pinch_armed
        self._middle_pinch_prev = self._right_pinch_armed
        self._dual_pinch_prev = self._dual_pinch_active

    def _handle_clicks_two_hand(
        self,
        secondary_hand: HandState | None,
        secondary_gesture: HandGestureState | None,
        actions: list[str],
    ) -> None:
        if secondary_hand is None or secondary_gesture is None:
            self._reset_pinch_state()
            if self._drag_active:
                self._mouse.release(Button.left)
                self._drag_active = False
                self._remember_action("Drag end")
            return

        side = self._normalize_side(secondary_hand.side)
        if side != "Left":
            self._reset_pinch_state()
            if self._drag_active:
                self._mouse.release(Button.left)
                self._drag_active = False
                self._remember_action("Drag end")
            return

        self._handle_clicks_pinch(secondary_gesture, actions, freeze_cursor=True)

    def _handle_clicks_dwell(
        self,
        primary_gesture: HandGestureState,
        actions: list[str],
    ) -> None:
        self._reset_pinch_state()
        if self._drag_active:
            self._mouse.release(Button.left)
            self._drag_active = False
            self._remember_action("Drag end")

        now = time.monotonic()
        if self._filtered_cursor is None:
            self._dwell_anchor = None
            self._dwell_start = 0.0
            self._dwell_fired = False
            return

        current = self._filtered_cursor.copy()
        if self._dwell_anchor is None:
            self._dwell_anchor = current
            self._dwell_start = now
            self._dwell_fired = False
            return

        motion = float(np.linalg.norm(current - self._dwell_anchor))
        if motion > float(self._config.dwell_rearm_move_px):
            self._dwell_anchor = current
            self._dwell_start = now
            self._dwell_fired = False
            return

        if primary_gesture.fist:
            self._dwell_anchor = current
            self._dwell_start = now
            self._dwell_fired = False
            return

        if motion > float(self._config.dwell_radius_px):
            self._dwell_start = now
            self._dwell_fired = False
            return

        if not self._dwell_fired and (now - self._dwell_start) >= float(self._config.dwell_click_sec):
            if self._check_cooldown("dwell_click", now, self._config.click_cooldown_sec):
                self._mouse.click(Button.left, 1)
                self._note_action(actions, "Dwell left click")
                self._dwell_fired = True
                self._cursor_lock_until = max(
                    self._cursor_lock_until,
                    now + float(self._config.click_cursor_lock_sec),
                )

    def _handle_shortcuts(
        self,
        hand: HandState,
        gesture: HandGestureState,
        actions: list[str],
    ) -> None:
        logical_side = self._normalize_side(hand.side)
        prefix = logical_side.lower()
        desktop_ready = self._desktop_gesture_ready(hand, gesture)

        self._trigger_shortcut(
            token=f"{prefix}:peace",
            active=gesture.peace_sign,
            label="Back",
            actions=actions,
            callback=lambda: self._tap_combo([Key.alt_l, Key.left]),
        )
        self._trigger_shortcut(
            token=f"{prefix}:three",
            active=gesture.three_fingers,
            label="Forward",
            actions=actions,
            callback=lambda: self._tap_combo([Key.alt_l, Key.right]),
        )
        self._trigger_shortcut(
            token=f"{prefix}:thumbs_up",
            active=gesture.thumbs_up,
            label="Play/Pause",
            actions=actions,
            callback=self._tap_play_pause,
        )
        self._trigger_shortcut(
            token=f"{prefix}:desktop",
            active=desktop_ready,
            label="Show desktop",
            actions=actions,
            callback=self._trigger_show_desktop,
        )

    def _trigger_shortcut(
        self,
        token: str,
        active: bool,
        label: str,
        actions: list[str],
        callback: Callable[[], None],
    ) -> None:
        was_active = self._gesture_prev.get(token, False)
        self._gesture_prev[token] = active
        if not active or was_active:
            return

        now = time.monotonic()
        if not self._check_cooldown(token, now, self._config.shortcut_cooldown_sec):
            return

        callback()
        self._note_action(actions, f"Shortcut {label}")

    def _check_cooldown(self, token: str, now: float, cooldown: float) -> bool:
        last_time = self._gesture_last_fire.get(token, -10.0)
        if now - last_time < cooldown:
            return False
        self._gesture_last_fire[token] = now
        return True

    def _tap_key(self, key) -> None:
        self._keyboard.press(key)
        self._keyboard.release(key)

    def _tap_combo(self, keys: list[object]) -> None:
        for key in keys:
            self._keyboard.press(key)
        for key in reversed(keys):
            self._keyboard.release(key)

    def _tap_play_pause(self) -> None:
        media_key = getattr(Key, "media_play_pause", None)
        if media_key is not None:
            self._tap_key(media_key)
        else:
            self._tap_key(Key.space)

    def _trigger_show_desktop(self) -> None:
        if sys.platform.startswith("linux"):
            self._tap_combo([Key.ctrl_l, Key.alt_l, "d"])
            return
        self._tap_combo([Key.cmd, "d"])

    def _note_action(self, actions: list[str], text: str) -> None:
        actions.append(text)
        self._remember_action(text)

    def _remember_action(self, text: str) -> None:
        self._last_action = text
        self._last_action_time = time.monotonic()
