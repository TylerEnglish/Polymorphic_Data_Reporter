from __future__ import annotations
import json, logging, sys
from typing import Any, Dict

class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        # attach extra if present
        for k, v in getattr(record, "__dict__", {}).items():
            if k not in ("name", "msg", "args", "levelname", "levelno",
                         "pathname", "filename", "module", "exc_info",
                         "exc_text", "stack_info", "lineno", "funcName",
                         "created", "msecs", "relativeCreated", "thread",
                         "threadName", "processName", "process"):
                payload[k] = v
        return json.dumps(payload, separators=(",", ":"))

def get_logger(name: str = "polymorphic", level: str = "INFO", structured_json: bool = True) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    handler = logging.StreamHandler(sys.stdout)
    if structured_json:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger