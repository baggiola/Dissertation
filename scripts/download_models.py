"""Download the MediaPipe pose landmarker model files into models/.

Run once after cloning:  python scripts/download_models.py
"""

from pathlib import Path
from urllib.request import urlretrieve

BASE_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker"
VARIANTS = ("lite", "full", "heavy")
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    for variant in VARIANTS:
        name = f"pose_landmarker_{variant}.task"
        target = MODELS_DIR / name
        if target.exists():
            print(f"{name}: already present")
            continue
        url = f"{BASE_URL}/pose_landmarker_{variant}/float16/latest/{name}"
        print(f"{name}: downloading...")
        urlretrieve(url, target)
        print(f"{name}: {target.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
