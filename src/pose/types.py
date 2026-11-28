"""Backend-independent pose data structures.

Everything downstream of pose estimation (angles, smoothing, comparison,
feedback) depends only on these types, never on a specific library.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import IntEnum


class Joint(IntEnum):
    """Named body landmarks.

    Values follow the 33-point BlazePose topology used by MediaPipe, which
    keeps that backend's conversion trivial. Other backends may provide only
    a subset; consumers must handle missing joints.
    """

    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32


# Body skeleton for drawing (face detail and hands left out on purpose).
SKELETON: tuple[tuple[Joint, Joint], ...] = (
    (Joint.LEFT_SHOULDER, Joint.RIGHT_SHOULDER),
    (Joint.LEFT_SHOULDER, Joint.LEFT_ELBOW),
    (Joint.LEFT_ELBOW, Joint.LEFT_WRIST),
    (Joint.RIGHT_SHOULDER, Joint.RIGHT_ELBOW),
    (Joint.RIGHT_ELBOW, Joint.RIGHT_WRIST),
    (Joint.LEFT_SHOULDER, Joint.LEFT_HIP),
    (Joint.RIGHT_SHOULDER, Joint.RIGHT_HIP),
    (Joint.LEFT_HIP, Joint.RIGHT_HIP),
    (Joint.LEFT_HIP, Joint.LEFT_KNEE),
    (Joint.LEFT_KNEE, Joint.LEFT_ANKLE),
    (Joint.RIGHT_HIP, Joint.RIGHT_KNEE),
    (Joint.RIGHT_KNEE, Joint.RIGHT_ANKLE),
    (Joint.LEFT_ANKLE, Joint.LEFT_HEEL),
    (Joint.LEFT_HEEL, Joint.LEFT_FOOT_INDEX),
    (Joint.RIGHT_ANKLE, Joint.RIGHT_HEEL),
    (Joint.RIGHT_HEEL, Joint.RIGHT_FOOT_INDEX),
)


@dataclass(frozen=True, slots=True)
class Landmark:
    """A single landmark position.

    For image landmarks, x and y are normalised to [0, 1] across the frame
    width and height and z is relative depth. For world landmarks, all three
    are in metres relative to the hip centre.
    """

    x: float
    y: float
    z: float
    visibility: float = 1.0


@dataclass(frozen=True, slots=True)
class PoseFrame:
    """One person's pose in one video frame."""

    timestamp_ms: int
    image: Mapping[Joint, Landmark]
    world: Mapping[Joint, Landmark] | None = None
