"""Constants for hand tracking, bones, and mesh topology."""

from __future__ import annotations

BONE_CONNECTIONS: list[tuple[int, int]] = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
]

FINGER_JOINT_TRIPLETS: dict[str, list[tuple[int, int, int]]] = {
    "Thumb": [(1, 2, 3), (2, 3, 4)],
    "Index": [(5, 6, 7), (6, 7, 8)],
    "Middle": [(9, 10, 11), (10, 11, 12)],
    "Ring": [(13, 14, 15), (14, 15, 16)],
    "Pinky": [(17, 18, 19), (18, 19, 20)],
}

FINGER_ORDER: list[str] = ["Thumb", "Index", "Middle", "Ring", "Pinky"]

HAND_SIDE_ORDER: list[str] = ["Left", "Right"]

# Fallback triangles if dynamic triangulation fails.
FALLBACK_HAND_FACES: list[tuple[int, int, int]] = [
    (0, 1, 5),
    (0, 5, 9),
    (0, 9, 13),
    (0, 13, 17),
    (1, 2, 5),
    (2, 3, 6),
    (3, 4, 7),
    (5, 6, 9),
    (6, 7, 10),
    (7, 8, 11),
    (9, 10, 13),
    (10, 11, 14),
    (11, 12, 15),
    (13, 14, 17),
    (14, 15, 18),
    (15, 16, 19),
    (17, 18, 19),
    (18, 19, 20),
]
