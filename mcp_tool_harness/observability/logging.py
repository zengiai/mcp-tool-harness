from __future__ import annotations

import json
import logging as std_logging
import time
from typing import Any


_STANDARD_ATTRS = {
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "module",
    "msecs",
    "message",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
}


class JsonFormatter(std_logging.Formatter):
    def format(self, record: std_logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        span = _current_span()
        if span is not None:
            payload["trace_id"] = span.context.trace_id
            payload["span_id"] = span.context.span_id
        for key, value in record.__dict__.items():
            if key not in _STANDARD_ATTRS and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


class TraceContextFilter(std_logging.Filter):
    def filter(self, record: std_logging.LogRecord) -> bool:
        span = _current_span()
        record.trace_id = span.context.trace_id if span is not None else None
        record.span_id = span.context.span_id if span is not None else None
        return True


def configure_logging(
    *,
    level: int | str = std_logging.INFO,
    json_format: bool = True,
    logger_name: str | None = None,
) -> std_logging.Logger:
    logger = std_logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.handlers.clear()
    handler = std_logging.StreamHandler()
    if json_format:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(std_logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
    handler.addFilter(TraceContextFilter())
    logger.addHandler(handler)
    logger.propagate = logger_name is not None
    return logger


def get_logger(name: str | None = None) -> std_logging.Logger:
    return std_logging.getLogger(name)


def _current_span():
    try:
        from .tracing import current_span

        return current_span()
    except Exception:
        return None


__all__ = ["JsonFormatter", "TraceContextFilter", "configure_logging", "get_logger"]
