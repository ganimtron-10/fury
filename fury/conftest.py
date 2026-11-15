import functools
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import os
import threading
import warnings

import pytest

warnings.simplefilter(action="default", category=FutureWarning)


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session, exitstatus):
    have_werrors = os.getenv("FURY_WERRORS", False)
    if have_werrors:
        # Check if there were any warnings during the test session
        reporter = session.config.pluginmanager.get_plugin("terminalreporter")
        if reporter.stats.get("warnings", None):
            session.exitstatus = 2


@pytest.hookimpl
def pytest_terminal_summary(terminalreporter, exitstatus, config):
    have_werrors = os.getenv("FURY_WERRORS", False)
    have_warnings = terminalreporter.stats.get("warnings", None)
    if have_warnings and have_werrors:
        terminalreporter.ensure_newline()
        terminalreporter.section("Werrors", sep="=", red=True, bold=True)
        terminalreporter.line(
            "Warnings as errors: Activated. \n"
            f"{len(have_warnings)} warnings were raised and "
            "treated as errors. \n"
        )


class _RecordingRequestHandler(SimpleHTTPRequestHandler):
    """Serve files from a folder, record each requested path, log nothing."""

    def do_GET(self):
        """Record the requested path, then serve the file."""
        self.server.requested_paths.append(self.path)
        super().do_GET()

    def log_message(self, *args):
        """
        Keep the test output free of request logs.

        Parameters
        ----------
        *args : tuple
            Format string and values of the log message, ignored.
        """


@pytest.fixture
def local_http_server(tmp_path):
    """
    Serve a temporary folder over HTTP on localhost.

    Yields
    ------
    ThreadingHTTPServer
        The running server. ``server.url`` is its address with a trailing
        slash, ``server.folder`` the served folder and
        ``server.requested_paths`` the paths requested so far.
    """
    folder = tmp_path / "served"
    folder.mkdir()
    handler = functools.partial(_RecordingRequestHandler, directory=str(folder))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    server.folder = folder
    server.requested_paths = []
    server.url = f"http://127.0.0.1:{server.server_address[1]}/"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
