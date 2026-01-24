import xml.etree.ElementTree as ET

import pytest

from fury.network.helpers import (
    enforce_type,
    find_by_tag,
    get_rgb_string,
    get_tag_name,
    infer_gexf_type,
    parse_rgb_string,
)


def test_get_rgb_string_normalized():
    # Case: Floats 0.0-1.0
    # 1.0 * 255 = 255
    assert get_rgb_string([1.0, 0.0, 0.0]) == "rgb(255,0,0)"
    assert get_rgb_string([0.0, 1.0, 0.5]) == "rgb(0,255,127)"


def test_get_rgb_string_raw():
    # Case: Integers/Floats > 1.0 (Already 0-255)
    assert get_rgb_string([255, 0, 100]) == "rgb(255,0,100)"
    assert get_rgb_string([255.0, 0.0, 100.0]) == "rgb(255,0,100)"


def test_get_rgb_string_alpha():
    # Case: RGBA
    # Note: Your logic converts alpha to 0-255 int if inputs are <= 1.0
    # Input: [0, 0, 0, 1] -> All <= 1.0 -> Scaled by 255 -> rgba(0,0,0,255)
    assert get_rgb_string([0.0, 0.0, 0.0, 1.0]) == "rgba(0,0,0,255)"

    # Case: Raw RGBA [255, 255, 255, 1] -> 255 > 1.0, so no scaling -> rgba(255,255,255,1)
    assert get_rgb_string([255, 255, 255, 1]) == "rgba(255,255,255,1)"


def test_get_rgb_string_empty():
    assert get_rgb_string([]) == "rgb(0,0,0)"
    assert get_rgb_string(None) == "rgb(0,0,0)"


def test_parse_rgb_string():
    # Basic parsing
    assert parse_rgb_string("rgb(255, 0, 0)") == [1.0, 0.0, 0.0]
    assert parse_rgb_string("rgba(0, 255, 0, 1.0)") == [0.0, 1.0, 0.0, 1.0]

    # Handling normalization (inputs > 1 are divided by 255)
    assert parse_rgb_string("rgb(255, 255, 255)") == [1.0, 1.0, 1.0]

    # Handling pre-normalized inputs
    assert parse_rgb_string("rgb(1.0, 0.0, 0.0)") == [1.0, 0.0, 0.0]


def test_parse_rgb_string_edge_cases():
    assert parse_rgb_string("") == [0.0, 0.0, 0.0]
    assert parse_rgb_string("invalid") == [0.0, 0.0, 0.0]
    # Only 2 numbers found
    assert parse_rgb_string("rgb(255, 255)") == [0.0, 0.0, 0.0]


def test_enforce_type():
    # Integers
    assert enforce_type("123", "integer") == 123
    assert enforce_type("123", "int") == 123
    assert enforce_type("invalid", "integer") == 0

    # Floats
    assert enforce_type("123.45", "float") == pytest.approx(123.45)
    assert enforce_type("invalid", "float") == 0.0

    # Booleans
    assert enforce_type("true", "boolean") is True
    assert enforce_type("True", "boolean") is True
    assert enforce_type("false", "boolean") is False
    assert enforce_type("random", "boolean") is False

    # List Strings
    assert enforce_type("a|b|c", "liststring") == ["a", "b", "c"]
    assert enforce_type("", "liststring") == []
    assert enforce_type(None, "liststring") is None


def test_infer_gexf_type():
    assert infer_gexf_type(True) == "boolean"
    assert infer_gexf_type(123) == "integer"
    assert infer_gexf_type(12.34) == "float"
    assert infer_gexf_type("text") == "string"


@pytest.fixture
def xml_fixture():
    # Create a mock XML structure with namespaces
    # <ns:root xmlns:ns="http://example.com">
    #   <ns:child>Content</ns:child>
    #   <other>Content</other>
    # </ns:root>
    root = ET.Element("{http://example.com}root")
    child_ns = ET.SubElement(root, "{http://example.com}child")
    child_no_ns = ET.SubElement(root, "other")
    return root, child_ns, child_no_ns


def test_get_tag_name(xml_fixture):
    root, child_ns, child_no_ns = xml_fixture
    assert get_tag_name(root) == "root"
    assert get_tag_name(child_ns) == "child"
    assert get_tag_name(child_no_ns) == "other"


def test_find_by_tag(xml_fixture):
    root, child_ns, child_no_ns = xml_fixture

    # Should find 'child' ignoring namespace
    results = find_by_tag(root, "child")
    assert len(results) == 1
    assert results[0] == child_ns

    # Should find 'other'
    results = find_by_tag(root, "other")
    assert len(results) == 1
    assert results[0] == child_no_ns

    # Should be case-insensitive (based on your implementation logic)
    results = find_by_tag(root, "CHILD")
    assert len(results) == 1
