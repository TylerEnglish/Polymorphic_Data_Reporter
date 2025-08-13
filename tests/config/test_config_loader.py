from pathlib import Path

def test_config_loads(cfg):
    # sanity top-level
    assert cfg.env.project_name
    assert cfg.storage.local.raw_root
    assert cfg.reports.enabled_generators.csv is True
    assert isinstance(cfg.charts.export_static_png, bool)

def test_paths_are_resolved_absolute(cfg):
    # after normalizer runs, local roots should be absolute paths
    assert Path(cfg.storage.local.raw_root).is_absolute()
    assert Path(cfg.storage.local.gold_root).is_absolute()

def test_outliers_detect_alias(cfg):
    # TOML uses [cleaning.outliers].detect, model maps to .method
    assert cfg.cleaning.outliers.method in ("zscore", "iqr")