import pytest

from aqlinreg.config import load_config


def test_default_config_loads():
    cfg = load_config("configs/default.yaml")
    assert cfg.seed == 42
    assert cfg.data.target == "NO2(GT)"
    assert len(cfg.features.all) == 15
    assert cfg.data.train_frac + cfg.data.val_frac < 1


def test_unknown_key_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "seed: 1\ndata: {url: x, typo_key: 1}\nfeatures: {raw: [a], time: []}\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="typo_key"):
        load_config(bad)
