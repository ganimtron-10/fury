import re


def get_rgb_string(values):
    """Converts list of [r, g, b, a] (0-1 or 0-255) to css string."""
    if not values:
        return "rgb(0,0,0)"

    # Heuristic: if values are floats <= 1.0, map to 0-255
    is_float_norm = all(v <= 1.0 for v in values)

    final_vals = []
    for v in values:
        val = int(v * 255) if is_float_norm else int(v)
        final_vals.append(str(val))

    if len(final_vals) > 3:
        return f"rgba({','.join(final_vals)})"
    return f"rgb({','.join(final_vals[:3])})"


def parse_rgb_string(color_str: str):
    """Parses rgb/rgba string to [r, g, b, a] normalized 0-1."""
    if not color_str:
        return [0.0, 0.0, 0.0]

    nums = re.findall(r"\d*\.?\d+", color_str)
    if len(nums) >= 3:
        res = [float(n) for n in nums]
        # Normalize if > 1.0 (assuming 8-bit color)
        if any(x > 1.0 for x in res[:3]):
            res = [x / 255.0 if i < 3 else x for i, x in enumerate(res)]
        return res
    return [0.0, 0.0, 0.0]


def enforce_type(value, target_type: str):
    """Enforces basic types based on string descriptors."""
    if value is None:
        return None

    target_type = target_type.lower()
    if target_type == "boolean":
        return str(value).lower() == "true"
    elif target_type in ["integer", "long", "int"]:
        try:
            return int(value)
        except (ValueError, TypeError):
            return 0
    elif target_type in ["float", "double"]:
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0
    elif target_type == "liststring":
        return value.split("|") if isinstance(value, str) and value != "" else []
    return value


def infer_gexf_type(value):
    """Infers GEXF attribute type from python value."""
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "float"
    return "string"


def get_tag_name(element):
    """Removes namespace {uri} from tag name."""
    if "}" in element.tag:
        return element.tag.split("}", 1)[1]
    return element.tag


def find_by_tag(node, tag_name):
    """Finds children ignoring namespace."""
    matches = []
    for child in node:
        if get_tag_name(child).lower() == tag_name.lower():
            matches.append(child)
    return matches
