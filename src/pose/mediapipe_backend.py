"""PoseEstimator backed by the MediaPipe Tasks PoseLandmarker.

This is the only module that imports MediaPipe.
"""

from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

from trainer.pose.types import Joint, Landmark, PoseFrame


class MediaPipePoseEstimator:
    """Runs PoseLandmarker in VIDEO mode: synchronous, with tracking between frames."""

    def __init__(
        self,
        model_path: Path,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Pose model not found at {model_path}. Run: python scripts/download_models.py"
            )
        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        self._last_timestamp_ms = -1

    def estimate(self, frame_bgr: np.ndarray, timestamp_ms: int) -> PoseFrame | None:
        # VIDEO mode rejects timestamps that do not strictly increase.
        timestamp_ms = max(timestamp_ms, self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, timestamp_ms)

        if not result.pose_landmarks:
            return None
        world = _convert(result.pose_world_landmarks[0]) if result.pose_world_landmarks else None
        return PoseFrame(
            timestamp_ms=timestamp_ms,
            image=_convert(result.pose_landmarks[0]),
            world=world,
        )

    def close(self) -> None:
        self._landmarker.close()


def _convert(landmarks: list) -> dict[Joint, Landmark]:
    return {
        Joint(i): Landmark(lm.x, lm.y, lm.z, lm.visibility if lm.visibility is not None else 1.0)
        for i, lm in enumerate(landmarks)
    }
