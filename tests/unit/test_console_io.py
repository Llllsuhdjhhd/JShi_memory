import logging

from rems.observability.console_io import configure_logging_stdout, configure_stdio_utf8


def test_configure_stdio_utf8_does_not_raise():
    configure_stdio_utf8()


def test_configure_logging_stdout_does_not_raise(capsys):
    configure_logging_stdout()
    logging.getLogger("test").warning("测试日志")
