import json
import logging

from groundwork.logs import JsonFormatter


def _record(message: str, **extra: object) -> logging.LogRecord:
    record = logging.LogRecord("groundwork.test", logging.WARNING, __file__, 1, message, None, None)
    record.__dict__.update(extra)
    return record


def test_event_level_logger_and_utc_time() -> None:
    payload = json.loads(JsonFormatter().format(_record("corpus verified")))
    assert payload["event"] == "corpus verified"
    assert payload["level"] == "WARNING"
    assert payload["logger"] == "groundwork.test"
    assert payload["time"].endswith("+00:00")


def test_extra_fields_become_top_level_keys() -> None:
    payload = json.loads(
        JsonFormatter().format(_record("done", queries=300, corpus="natural-questions"))
    )
    assert payload["queries"] == 300
    assert payload["corpus"] == "natural-questions"


def test_standard_record_attributes_are_not_repeated() -> None:
    payload = json.loads(JsonFormatter().format(_record("done")))
    assert "msg" not in payload
    assert "lineno" not in payload


def test_exception_traceback_is_included() -> None:
    error = ValueError("boom")
    record = _record("failed")
    record.exc_info = (ValueError, error, error.__traceback__)
    payload = json.loads(JsonFormatter().format(record))
    assert "ValueError: boom" in payload["exception"]


def test_values_that_are_not_json_serialisable_are_rendered_as_text() -> None:
    payload = json.loads(JsonFormatter().format(_record("done", path=object())))
    assert payload["path"].startswith("<object object")
