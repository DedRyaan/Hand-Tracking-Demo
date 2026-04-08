"""Gesture primitives for the Air Mouse controller."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hand_tracker.tracker import HandState


FLEX_OPEN_THRESHOLD = 30.0
FLEX_CLOSED_THRESHOLD = 65.0


@dataclass
class HandGestureState:
    side: str
    palm_scale_m: float
    index_thumb_ratio: float
    middle_thumb_ratio: float
    index_thumb_pinch: bool
    middle_thumb_pinch: bool
    thumb_open: bool
    index_open: bool
    middle_open: bool
    ring_open: bool
    pinky_open: bool
    fist: bool
    open_palm: bool
    peace_sign: bool
    three_fingers: bool
    thumbs_up: bool
    german_three_sign: bool


def index_tip_xy(hand: HandState) -> np.ndarray:
    return hand.landmarks[8, :2].astype(np.float32)


def palm_center_xy(hand: HandState) -> np.ndarray:
    center = np.mean(hand.landmarks[[0, 5, 9, 13, 17], :2], axis=0)
    return center.astype(np.float32)


def _is_open(flexion: float) -> bool:
    return float(flexion) < FLEX_OPEN_THRESHOLD


def _is_closed(flexion: float) -> bool:
    return float(flexion) > FLEX_CLOSED_THRESHOLD


def _pinch_ratio(hand: HandState, first_idx: int, second_idx: int) -> float:
    palm_scale = max(0.04, float(np.linalg.norm(hand.camera_vertices[5] - hand.camera_vertices[17])))
    pinch_distance = float(np.linalg.norm(hand.camera_vertices[first_idx] - hand.camera_vertices[second_idx]))
    return pinch_distance / palm_scale


def classify_hand_gesture(hand: HandState) -> HandGestureState:
    thumb_flex = float(hand.finger_flexion.get("Thumb", 90.0))
    index_flex = float(hand.finger_flexion.get("Index", 90.0))
    middle_flex = float(hand.finger_flexion.get("Middle", 90.0))
    ring_flex = float(hand.finger_flexion.get("Ring", 90.0))
    pinky_flex = float(hand.finger_flexion.get("Pinky", 90.0))

    thumb_open = _is_open(thumb_flex)
    index_open = _is_open(index_flex)
    middle_open = _is_open(middle_flex)
    ring_open = _is_open(ring_flex)
    pinky_open = _is_open(pinky_flex)

    index_closed = _is_closed(index_flex)
    middle_closed = _is_closed(middle_flex)
    ring_closed = _is_closed(ring_flex)
    pinky_closed = _is_closed(pinky_flex)

    palm_scale_m = max(0.04, float(np.linalg.norm(hand.camera_vertices[5] - hand.camera_vertices[17])))
    open_palm = thumb_open and index_open and middle_open and ring_open and pinky_open
    index_ratio = _pinch_ratio(hand, 4, 8)
    middle_ratio = _pinch_ratio(hand, 4, 12)

    thumb_closed = _is_closed(thumb_flex)

    german_three_sign = (
        thumb_open
        and index_open
        and middle_open
        and ring_closed
        and pinky_closed
    )
    if german_three_sign and thumb_closed:
        german_three_sign = False

    return HandGestureState(
        side=hand.side,
        palm_scale_m=palm_scale_m,
        index_thumb_ratio=index_ratio,
        middle_thumb_ratio=middle_ratio,
        index_thumb_pinch=index_ratio < 0.56,
        middle_thumb_pinch=middle_ratio < 0.58,
        thumb_open=thumb_open,
        index_open=index_open,
        middle_open=middle_open,
        ring_open=ring_open,
        pinky_open=pinky_open,
        fist=index_closed and middle_closed and ring_closed and pinky_closed,
        open_palm=open_palm,
        peace_sign=index_open and middle_open and ring_closed and pinky_closed,
        three_fingers=index_open and middle_open and ring_open and pinky_closed,
        thumbs_up=thumb_open and index_closed and middle_closed and ring_closed and pinky_closed,
        german_three_sign=german_three_sign,
    )
