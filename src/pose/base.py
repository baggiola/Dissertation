"""The interface every pose backend implements."""

from typing import Protocol

import numpy as np

from trainer.pose.types import PoseFrame


class PoseEstimator(Protocol):
    """Estimates a single person's pose from video frames."""

    def estimate(self, frame_bgr: np.ndarray, timestamp_ms: int) -> PoseFrame | None:
        """Return the pose in ``frame_bgr`` (OpenCV BGR order), or None if no person is found.

        ``timestamp_ms`` should increase from call to call; backends that track
        across frames rely on it.
        """
        ...

    def close(self) -> None:
        """Release any resources held by the backend."""
        ...
