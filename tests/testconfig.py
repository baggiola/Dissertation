from pathlib import Path

from trainer.config import DEFAULT_PATH, load_config


def test_default_config_loads() -> None:
    config = load_config(local_path=None)
    assert config["camera"]["index"] == 0
    assert config["smoothing"]["method"] in {"one_euro", "ema", "none"}


def test_local_overrides_merge(tmp_path: Path) -> None:
    local = tmp_path / "local.toml"
    local.write_text('[haptics]\nport = "COM7"\n')
    config = load_config(DEFAULT_PATH, local)
    assert config["haptics"]["port"] == "COM7"
    assert config["haptics"]["baud"] == 115200  # untouched keys survive
