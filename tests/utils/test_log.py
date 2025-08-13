import json
import logging
from src.utils.log import get_logger, JsonFormatter

def test_json_formatter_basic_and_extra():
    logger = logging.getLogger("t-json")
    logger.handlers.clear()
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    # capture a single record via a proper Handler
    class CapHandler(logging.Handler):
        def __init__(self):
            super().__init__(level=0)
            self.last = None
            self._formatter = handler.formatter  # reuse the JsonFormatter

        def emit(self, record: logging.LogRecord) -> None:
            self.last = self._formatter.format(record)

    cap = CapHandler()
    logger.handlers = [cap]  # hijack with our capturing handler

    logger.info("hello", extra={"dataset_slug": "demo"})
    payload = json.loads(cap.last)
    assert payload["message"] == "hello"
    assert payload["level"] == "INFO"
    assert payload["dataset_slug"] == "demo"
    assert "time" in payload

def test_get_logger_idempotent_and_plain_mode():
    lg1 = get_logger("poly-test", level="DEBUG", structured_json=True)
    lg2 = get_logger("poly-test", level="INFO", structured_json=True)
    assert lg1 is lg2
    lg3 = get_logger("poly-plain", level="INFO", structured_json=False)
    assert isinstance(lg3.handlers[0].formatter, logging.Formatter)