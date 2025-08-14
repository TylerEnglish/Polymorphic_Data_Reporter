from __future__ import annotations

try:
    from ..config_model.model import RootCfg
    _cfg = RootCfg.load()
    INVENTORY_KEY = _cfg.nlg.inventory_key
    NARRATIVE_FILENAME = _cfg.nlg.narrative_filename
except Exception:
    # Fallbacks if config cannot be loaded (tests, minimal environments, etc.)
    INVENTORY_KEY = "_inventory"
    NARRATIVE_FILENAME = "narrative.txt"
