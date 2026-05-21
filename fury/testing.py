"""Utilities for testing."""

from contextlib import contextmanager
from distutils.version import LooseVersion
from functools import partial
import io
import json
import operator
import sys
import warnings

import numpy as np
from numpy.testing import assert_array_equal
import scipy  # type: ignore


@contextmanager
def captured_output():
    """Capture stdout and stderr from print or logging.

    This context manager temporarily replaces sys.stdout and sys.stderr
    to capture printed output and return it for testing.

    Returns
    -------
    out : StringIO
        Object containing captured stdout.
    err : StringIO
        Object containing captured stderr.

    Examples
    --------
    >>> def foo():
    ...    print('hello world!')
    >>> with captured_output() as (out, err):
    ...    foo()
    >>> print(out.getvalue().strip())
    hello world!
    """
    new_out, new_err = io.StringIO(), io.StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = new_out, new_err
        yield sys.stdout, sys.stderr
    finally:
        sys.stdout, sys.stderr = old_out, old_err


def assert_operator(value1, value2, *, msg="", op=operator.eq):
    """Check boolean statement using the given operator.

    Compares two values using the provided operator and raises
    an AssertionError if the comparison is false.

    Parameters
    ----------
    value1 : object
        First value to be compared.
    value2 : object
        Second value to be compared.
    msg : str, optional
        Error message to be displayed if the assertion fails.
        Can contain format placeholders for values.
    op : callable, optional
        Operator to compare values. Default is equality operator.

    Raises
    ------
    AssertionError
        If the comparison between value1 and value2 using op returns False.
    """
    if not op(value1, value2):
        raise AssertionError(msg.format(str(value2), str(value1)))


assert_greater_equal = partial(
    assert_operator,
    op=operator.ge,
    msg="{0} >= {1}",
)
assert_greater = partial(assert_operator, op=operator.gt, msg="{0} > {1}")
assert_less_equal = partial(assert_operator, op=operator.le, msg="{0} =< {1}")
assert_less = partial(assert_operator, op=operator.lt, msg="{0} < {1}")
assert_true = partial(
    assert_operator, value2=True, op=operator.eq, msg="False is not true"
)
assert_false = partial(
    assert_operator, value2=False, op=operator.eq, msg="True is not false"
)
assert_not_equal = partial(assert_operator, op=operator.ne)
assert_equal = partial(assert_operator, op=operator.eq)


def assert_arrays_equal(arrays1, arrays2):
    """Check that all arrays in arrays1 equal the corresponding arrays in arrays2.

    Parameters
    ----------
    arrays1 : sequence of ndarray
        First sequence of arrays to be compared.
    arrays2 : sequence of ndarray
        Second sequence of arrays to be compared.

    Raises
    ------
    AssertionError
        If any corresponding arrays are not equal.
    """
    for arr1, arr2 in zip(arrays1, arrays2, strict=False):
        assert_array_equal(arr1, arr2)


class EventCounter:
    """Count and record UI events for testing.

    This class provides functionality to count event occurrences for UI testing
    and verification. It can record counts, save them to a file, and compare them
    with expected counts.

    Parameters
    ----------
    events_names : list of str, optional
        List of event names to count. If None, defaults to common VTK events.
    """

    def __init__(self, *, events_names=None):
        """Initialize the EventCounter.

        Parameters
        ----------
        events_names : list of str, optional
            List of event names to count. If None, defaults to common PyGfx events.
        """
        if events_names is None:
            events_names = [
                "pointer_down",
                "pointer_up",
                "pointer_move",
                "pointer_drag",
                "key_down",
                "key_up",
                "click",
                "double_click",
                "wheel",
            ]

        # Events to count
        self.events_counts = dict.fromkeys(events_names, 0)

    def count(self, i_ren, _obj, _element):
        """Count events occurrences.

        Parameters
        ----------
        i_ren : object
            The interaction renderer with event data.
        _obj : object
            The object that received the event.
        _element : object
            UI element that received the event.
        """
        self.events_counts[i_ren.event.name] += 1

    def monitor(self, ui_component):
        """Add callbacks to monitor events on a UI component.

        Parameters
        ----------
        ui_component : object
            UI component with actors to monitor for events.
        """
        for event in self.events_counts:
            for obj_actor in ui_component.actors:
                ui_component.add_callback(obj_actor, event, self.count)

    def save(self, filename):
        """Save event counts to a JSON file.

        Parameters
        ----------
        filename : str
            Path to save the event counts.
        """
        with open(filename, "w") as f:
            json.dump(self.events_counts, f)

    @classmethod
    def load(cls, filename):
        """Load event counts from a JSON file.

        Parameters
        ----------
        filename : str
            Path to the JSON file with saved event counts.

        Returns
        -------
        EventCounter
            A new EventCounter instance with loaded counts.
        """
        event_counter = cls()
        with open(filename) as f:
            event_counter.events_counts = json.load(f)

        return event_counter

    def check_counts(self, expected):
        """Compare current event counts with expected counts.

        Parameters
        ----------
        expected : EventCounter
            EventCounter instance with expected event counts.

        Raises
        ------
        AssertionError
            If the counts don't match the expected counts.
        """
        assert_equal(len(self.events_counts), len(expected.events_counts))

        # Useful loop for debugging.
        msg = "{}: {} vs. {} (expected)"
        for event, count in expected.events_counts.items():
            if self.events_counts[event] != count:
                print(msg.format(event, self.events_counts[event], count))

        msg = "Wrong count for '{}'."
        for event, count in expected.events_counts.items():
            assert_equal(
                self.events_counts[event],
                count,
                msg=msg.format(event),
            )


class clear_and_catch_warnings(warnings.catch_warnings):
    """Context manager that resets warning registry for catching warnings.

    Warnings can be slippery, because whenever a warning is triggered, Python
    adds a ``__warningregistry__`` member to the *calling* module. This makes
    it impossible to retrigger the warning in this module, whatever you put in
    the warnings filters. This context manager accepts a sequence of `modules`
    as a keyword argument to its constructor and:

    * stores and removes any ``__warningregistry__`` entries in given `modules`
      on entry;
    * resets ``__warningregistry__`` to its previous state on exit.

    This makes it possible to trigger any warning afresh inside the context
    manager without disturbing the state of warnings outside.

    Parameters
    ----------
    record : bool, optional
        Specifies whether warnings should be captured by a custom
        implementation of ``warnings.showwarning()`` and be appended to a list
        returned by the context manager. Otherwise None is returned by the
        context manager. Default is True.
    modules : sequence, optional
        Sequence of modules for which to reset warnings registry on entry and
        restore on exit.

    Notes
    -----
    This class is copied (with minor modifications) from the Nibabel package.
    https://github.com/nipy/nibabel. See COPYING file distributed along with
    the Nibabel package for the copyright and license terms.

    Examples
    --------
    >>> import warnings
    >>> with clear_and_catch_warnings(modules=[np.random.rand]):
    ...     warnings.simplefilter('always')
    ...     # do something that raises a warning in np.random.rand
    """

    class_modules = ()

    def __init__(self, *, record=True, modules=()):
        """Initialize the context manager.

        Parameters
        ----------
        record : bool, optional
            Specifies whether warnings should be captured by a custom
            implementation of ``warnings.showwarning()`` and be appended to a list
            returned by the context manager. Otherwise None is returned by the
            context manager. Default is True.
        modules : sequence, optional
            Sequence of modules for which to reset warnings registry on entry and
            restore on exit.
        """
        self.modules = set(modules).union(self.class_modules)
        self._warnreg_copies = {}
        super(clear_and_catch_warnings, self).__init__(record=record)

    def __enter__(self):
        """Clear warning registry for given modules.

        Returns
        -------
        clear_and_catch_warnings
            The context manager instance.
        """
        for mod in self.modules:
            if hasattr(mod, "__warningregistry__"):
                mod_reg = mod.__warningregistry__
                self._warnreg_copies[mod] = mod_reg.copy()
                mod_reg.clear()
        return super(clear_and_catch_warnings, self).__enter__()

    def __exit__(self, *exc_info):
        """Restore warning registry to its previous state.

        Parameters
        ----------
        *exc_info : tuple
            Exception information, if any, raised in the context.
        """
        super(clear_and_catch_warnings, self).__exit__(*exc_info)
        for mod in self.modules:
            if hasattr(mod, "__warningregistry__"):
                mod.__warningregistry__.clear()
            if mod in self._warnreg_copies:
                mod.__warningregistry__.update(self._warnreg_copies[mod])


def setup_test():
    """Set numpy print options to "legacy" for new versions of numpy.

    Configure numpy print options to maintain compatibility with older versions.
    If imported into a file, nosetest will run this before any doctests.

    References
    ----------
    https://github.com/numpy/numpy/commit/710e0327687b9f7653e5ac02d222ba62c657a718
    https://github.com/numpy/numpy/commit/734b907fc2f7af6e40ec989ca49ee6d87e21c495
    https://github.com/nipy/nibabel/pull/556
    """
    if LooseVersion(np.__version__) >= LooseVersion("1.14"):
        np.set_printoptions(legacy="1.13")

    # Temporary fix until scipy release in October 2018
    # must be removed after that
    # print the first occurrence of matching warnings for each location
    # (module + line number) where the warning is issued
    if (
        LooseVersion(np.__version__) >= LooseVersion("1.15")
        and LooseVersion(scipy.version.short_version) <= "1.1.0"
    ):
        warnings.simplefilter("default")


def check_for_warnings(warn_printed, w_msg):
    """Check for specific warnings in the warning registry.

    Parameters
    ----------
    warn_printed : list
        List of captured warnings.
    w_msg : str
        Warning message to check for.
    """
    selected_w = [w for w in warn_printed if issubclass(w.category, UserWarning)]
    assert len(selected_w) >= 1
    msg = [str(m.message) for m in selected_w]
    assert_equal(w_msg in msg, True)


def get_all_actors(scene):
    """Recursively traverse the main scene, overlays, and UI scene to gather all actors."""
    actors = []

    def traverse(obj):
        if obj is None:
            return
        if obj in actors:
            return

        if hasattr(obj, "add_event_handler"):
            actors.append(obj)

        if hasattr(obj, "children"):
            for child in obj.children:
                traverse(child)

    traverse(scene)
    if hasattr(scene, "main_scene"):
        traverse(scene.main_scene)
    if hasattr(scene, "ui_scene"):
        traverse(scene.ui_scene)

    return actors


def record_events(show_manager):
    """Record events during interaction."""
    if getattr(show_manager, "_events_recording_active", False):
        return

    show_manager._events_recording_active = True
    show_manager._recorded_events = []

    original_dispatch_event = show_manager.renderer.dispatch_event

    def wrapped_dispatch_event(event):
        if event.type in ["resize", "close", "focus", "blur", "pointer_enter", "pointer_leave"]:
            original_dispatch_event(event)
            return

        all_actors = get_all_actors(show_manager.scene)
        target_index = -1
        if event.target in all_actors:
            target_index = all_actors.index(event.target)

        event_dict = {
            "class": event.__class__.__name__,
            "type": event.type,
            "target_index": target_index,
        }
        for attr in ["x", "y", "button", "buttons", "dy", "dx", "key", "modifiers", "ntouch", "clicks"]:
            if hasattr(event, attr):
                event_dict[attr] = getattr(event, attr)

        show_manager._recorded_events.append(event_dict)
        original_dispatch_event(event)

    show_manager.renderer.dispatch_event = wrapped_dispatch_event


def record_events_to_file(show_manager, filename):
    """Record and save events to a file."""
    record_events(show_manager)
    show_manager.start()

    import gzip
    import json

    events_json = json.dumps(show_manager._recorded_events)
    if filename.endswith(".gz"):
        with gzip.open(filename, "wt", encoding="utf-8") as f:
            f.write(events_json)
    else:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(events_json)


def play_events(show_manager, events):
    """Play a sequence of recorded events."""
    show_manager.render()
    show_manager._draw_canvas()

    show_manager._playing_back = True
    try:
        from pygfx import PointerEvent, KeyboardEvent, Event

        all_actors = get_all_actors(show_manager.scene)

        for event_dict in events:
            cls_name = event_dict.get("class", "Event")
            evt_type = event_dict.get("type")
            target_index = event_dict.get("target_index", -1)

            kwargs = {}
            for attr in ["x", "y", "button", "buttons", "dy", "dx", "key", "modifiers", "ntouch", "clicks"]:
                if attr in event_dict:
                    kwargs[attr] = event_dict[attr]

            if cls_name == "PointerEvent":
                event = PointerEvent(type=evt_type, **kwargs)
            elif cls_name == "KeyboardEvent":
                event = KeyboardEvent(type=evt_type, **kwargs)
            else:
                event = Event(type=evt_type, **kwargs)

            target = None
            if target_index != -1 and target_index < len(all_actors):
                target = all_actors[target_index]
            else:
                target = show_manager.renderer

            path = []
            curr = target
            while curr is not None:
                if curr not in path:
                    path.append(curr)
                curr = getattr(curr, "parent", None)

            if show_manager.scene not in path:
                path.append(show_manager.scene)
            if show_manager.renderer not in path:
                path.append(show_manager.renderer)

            event._target = target
            for obj in path:
                if hasattr(obj, "handle_event"):
                    event._current_target = obj
                    obj.handle_event(event)
                    if getattr(event, "_propagation_stopped", False):
                        break

            show_manager._draw_canvas()

            # Add a small delay and poll events so that the user can visually see the simulation flow
            if not hasattr(show_manager.window, "draw"):  # not offscreen
                import time
                time.sleep(0.01)
                try:
                    import glfw
                    glfw.poll_events()
                except Exception:
                    pass
    finally:
        show_manager._playing_back = False


def play_events_from_file(show_manager, filename):
    """Load events from a file and replay them."""
    import gzip
    import json

    if filename.endswith(".gz"):
        with gzip.open(filename, "rt", encoding="utf-8") as f:
            events = json.loads(f.read())
    else:
        with open(filename, "r", encoding="utf-8") as f:
            events = json.loads(f.read())

    play_events(show_manager, events)
