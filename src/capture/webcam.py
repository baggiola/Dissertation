"""Webcam capture via OpenCV."""

import sys
import time
from types import TracebackType

import cv2
import numpy as np


class Webcam:
    """Reads BGR frames from a local camera, stamped with a monotonic time in ms."""

    def __init__(self, index: int = 0, width: int | None = None, height: int | None = None) -> None:
        # DirectShow opens much faster than the default MSMF backend on Windows.
        api = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        self._cap = cv2.VideoCapture(index, api)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open camera {index}")
        if width:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read(self) -> tuple[np.ndarray, int] | None:
        """Return ``(frame, timestamp_ms)``, or None if the camera gave no frame."""
        ok, frame = self._cap.read()
        if not ok:
            return None
        return frame, int(time.monotonic() * 1000)

    def close(self) -> None:
        self._cap.release()

    def __enter__(self) -> "Webcam":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
