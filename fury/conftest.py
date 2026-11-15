import os
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


@pytest.fixture(autouse=True)
def reset_ui_theme():
    """
    Start and end every test with the built-in default UI theme.

    The theme is set explicitly, so a theme set by one test does not leak
    into the next, and results do not depend on ``FURY_UI_THEME``.
    """
    from fury.ui import UIContext, builtin_theme
    from fury.ui.theme import _DEFAULT_THEME

    UIContext.theme = builtin_theme(_DEFAULT_THEME)
    yield
    UIContext.theme = builtin_theme(_DEFAULT_THEME)


@pytest.fixture
def classic_theme():
    """
    Use the classic global UI theme, for tests of the classic look.

    The theme is set without restyling elements left over from earlier
    tests.
    """
    from fury.ui import UIContext, builtin_theme

    UIContext.theme = builtin_theme("classic")
