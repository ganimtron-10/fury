import re
import xml.etree.ElementTree as ET

from fury.network.core import Edge, Network, Node
from fury.network.helpers import (
    enforce_type,
    find_by_tag,
    get_tag_name,
)


class BaseParser:
    def parse(self, data):
        raise NotImplementedError("Parse method not implemented")

    def stringify(self, network):
        raise NotImplementedError("Stringify method not implemented")


class GEXFParser(BaseParser):
    """Parses and Writes GEXF XML format."""

    def parse(self, xml_string):
        try:
            root = ET.fromstring(xml_string)
        except ET.ParseError as e:
            raise ValueError("Invalid GEXF XML Data") from e

        network = Network()

        # 1. Locate root elements
        gexf_root = root if get_tag_name(root).lower() == "gexf" else None
        if gexf_root is None:
            # Try to find it
            for child in root.iter():
                if get_tag_name(child).lower() == "gexf":
                    gexf_root = child
                    break

        if gexf_root is None:
            gexf_root = root  # Fallback

        graph_node = None
        for child in gexf_root:
            if get_tag_name(child).lower() == "graph":
                graph_node = child
                break

        if graph_node is None:
            raise ValueError("No <graph> tag found in GEXF")

        # 2. Metadata
        network.mode = graph_node.get("mode", "static")
        edge_type = graph_node.get("defaultedgetype", "undirected")
        network.directed = edge_type == "directed"

        meta_node = find_by_tag(gexf_root, "meta")
        if meta_node:
            for child in meta_node[0]:
                network.meta[get_tag_name(child)] = child.text

        # 3. Model (Attributes)
        attributes_nodes = find_by_tag(graph_node, "attributes")
        attr_definitions = {"node": {}, "edge": {}}  # ID -> {type, title}

        for attr_node in attributes_nodes:
            attr_class = attr_node.get("class", "node")
            for attr in find_by_tag(attr_node, "attribute"):
                a_id = attr.get("id")
                a_type = attr.get("type", "string")
                a_title = attr.get("title", a_id)
                attr_definitions[attr_class][a_id] = {"type": a_type, "title": a_title}
                # Update network model for re-export
                network.model[attr_class][a_title] = a_type

        # 4. Nodes
        nodes_node = find_by_tag(graph_node, "nodes")
        if nodes_node:
            for n in find_by_tag(nodes_node[0], "node"):
                n_id = n.get("id")
                n_label = n.get("label", "")
                node_obj = Node(n_id, n_label)

                # Attributes
                attvalues = find_by_tag(n, "attvalues")
                if attvalues:
                    for av in find_by_tag(attvalues[0], "attvalue"):
                        # GEXF uses 'for' or 'id' to reference the attribute definition
                        k_ref = av.get("for") or av.get("id")
                        val = av.get("value")

                        # Resolve type
                        if k_ref in attr_definitions["node"]:
                            def_entry = attr_definitions["node"][k_ref]
                            # Store using the human readable title if available, else ID
                            key_name = def_entry["title"]
                            node_obj.attributes[key_name] = enforce_type(
                                val, def_entry["type"]
                            )
                        else:
                            node_obj.attributes[k_ref] = val

                # Viz (Color, Position, Size)
                # Viz logic usually resides in specific namespace children
                for child in n:
                    tag = get_tag_name(child).lower()
                    if tag == "color":
                        r, g, b = child.get("r"), child.get("g"), child.get("b")
                        if r and g and b:
                            node_obj.viz["color"] = [
                                float(r) / 255.0,
                                float(g) / 255.0,
                                float(b) / 255.0,
                            ]
                    elif tag == "position":
                        x, y, z = child.get("x"), child.get("y"), child.get("z")
                        node_obj.viz["position"] = {
                            "x": float(x) if x else 0.0,
                            "y": float(y) if y else 0.0,
                            "z": float(z) if z else 0.0,
                        }
                    elif tag == "size":
                        node_obj.viz["size"] = float(child.get("value", 1.0))

                network.add_node(node_obj)

        # 5. Edges
        edges_node = find_by_tag(graph_node, "edges")
        if edges_node:
            for e in find_by_tag(edges_node[0], "edge"):
                e_id = e.get("id")
                source = e.get("source")
                target = e.get("target")
                weight = float(e.get("weight", 1.0))
                e_type = e.get("type", network.edge_type)

                edge_obj = Edge(e_id, source, target, weight, e_type)

                # Edge Attributes
                attvalues = find_by_tag(e, "attvalues")
                if attvalues:
                    for av in find_by_tag(attvalues[0], "attvalue"):
                        k_ref = av.get("for") or av.get("id")
                        val = av.get("value")

                        if k_ref in attr_definitions["edge"]:
                            def_entry = attr_definitions["edge"][k_ref]
                            key_name = def_entry["title"]
                            edge_obj.attributes[key_name] = enforce_type(
                                val, def_entry["type"]
                            )
                        else:
                            edge_obj.attributes[k_ref] = val

                network.add_edge(edge_obj)

        return network

    def stringify(self, network):
        # Create XML Structure
        gexf = ET.Element(
            "gexf",
            {
                "xmlns": "http://www.gexf.net/1.2draft",
                "xmlns:viz": "http://www.gexf.net/1.2draft/viz",
                "version": "1.2",
            },
        )

        meta = ET.SubElement(gexf, "meta")
        for k, v in network.meta.items():
            m = ET.SubElement(meta, k)
            m.text = str(v)

        graph = ET.SubElement(
            gexf,
            "graph",
            {"mode": network.mode, "defaultedgetype": network.edge_type},
        )

        # Generate Attributes Definitions (Model)
        # We need to map Attribute Keys to numeric IDs for GEXF
        attr_map = {"node": {}, "edge": {}}  # Name -> ID

        def create_attr_section(cls_name, schema):
            if not schema:
                return
            attrs_el = ET.SubElement(graph, "attributes", {"class": cls_name})
            for i, (name, type_str) in enumerate(schema.items()):
                attr_id = str(i)
                attr_map[cls_name][name] = attr_id
                ET.SubElement(
                    attrs_el,
                    "attribute",
                    {"id": attr_id, "title": name, "type": type_str},
                )

        create_attr_section("node", network.model.get("node"))
        create_attr_section("edge", network.model.get("edge"))

        # Nodes
        nodes_el = ET.SubElement(graph, "nodes")
        for node in network.nodes.values():
            n_el = ET.SubElement(nodes_el, "node", {"id": node.id, "label": node.label})

            # Attributes
            if node.attributes:
                avs_el = ET.SubElement(n_el, "attvalues")
                for k, v in node.attributes.items():
                    if k in attr_map["node"]:
                        ET.SubElement(
                            avs_el,
                            "attvalue",
                            {"for": attr_map["node"][k], "value": str(v)},
                        )

            # Viz
            if "color" in node.viz:
                c = node.viz["color"]  # assumes [r,g,b] 0-1
                ET.SubElement(
                    n_el,
                    "viz:color",
                    {
                        "r": str(int(c[0] * 255)),
                        "g": str(int(c[1] * 255)),
                        "b": str(int(c[2] * 255)),
                    },
                )
            if "position" in node.viz:
                p = node.viz["position"]
                ET.SubElement(
                    n_el,
                    "viz:position",
                    {
                        "x": str(p.get("x", 0)),
                        "y": str(p.get("y", 0)),
                        "z": str(p.get("z", 0)),
                    },
                )
            if "size" in node.viz:
                ET.SubElement(n_el, "viz:size", {"value": str(node.viz["size"])})

        # Edges
        edges_el = ET.SubElement(graph, "edges")
        for edge in network.edges:
            e_el = ET.SubElement(
                edges_el,
                "edge",
                {
                    "id": edge.id,
                    "source": edge.source,
                    "target": edge.target,
                    "weight": str(edge.weight),
                },
            )
            if edge.type != network.edge_type:
                e_el.set("type", edge.type)

            if edge.attributes:
                avs_el = ET.SubElement(e_el, "attvalues")
                for k, v in edge.attributes.items():
                    if k in attr_map["edge"]:
                        ET.SubElement(
                            avs_el,
                            "attvalue",
                            {"for": attr_map["edge"][k], "value": str(v)},
                        )

        # Return pretty string
        # Minidom is used for pretty printing, but ElementTree is safer for construction
        from xml.dom import minidom

        raw_str = ET.tostring(gexf, encoding="utf-8")
        return minidom.parseString(raw_str).toprettyxml(indent="  ")


class GMLParser(BaseParser):
    """Parses and Writes GML (Graph Modeling Language)."""

    def parse(self, data):
        # A simple stack-based tokenizer/parser for GML
        # GML structure: key [ ... ] or key value

        # 1. Tokenize
        tokens = []
        in_quote = False
        current_token = []

        for char in data:
            if char == '"':
                in_quote = not in_quote
                current_token.append(char)
            elif char.isspace() and not in_quote:
                if current_token:
                    tokens.append("".join(current_token))
                    current_token = []
            elif char in ["[", "]"] and not in_quote:
                if current_token:
                    tokens.append("".join(current_token))
                    current_token = []
                tokens.append(char)
            else:
                current_token.append(char)
        if current_token:
            tokens.append("".join(current_token))

        # 2. Parse into Hierarchy
        def parse_gml_level(token_iter):
            obj = {}
            while True:
                try:
                    key = next(token_iter)
                except StopIteration:
                    break

                if key == "]":
                    return obj

                try:
                    value_token = next(token_iter)
                except StopIteration:
                    break

                if value_token == "[":
                    val = parse_gml_level(token_iter)
                    # GML allows duplicate keys (e.g. node, edge). Store as list.
                    if key in obj:
                        if not isinstance(obj[key], list):
                            obj[key] = [obj[key]]
                        obj[key].append(val)
                    else:
                        obj[key] = (
                            val  # Single object first, convert to list if duplicate found
                        )
                else:
                    # Clean value
                    if value_token.startswith('"') and value_token.endswith('"'):
                        val = value_token[1:-1]
                    elif "." in value_token:
                        try:
                            val = float(value_token)
                        except:
                            val = value_token
                    else:
                        try:
                            val = int(value_token)
                        except:
                            val = value_token

                    obj[key] = val
            return obj

        iter_tokens = iter(tokens)
        parsed_root = parse_gml_level(iter_tokens)

        # 3. Convert to Network
        network = Network()
        if "graph" not in parsed_root:
            raise ValueError("GML must contain a 'graph' key")

        g_data = parsed_root["graph"]
        # Handle case where graph might be a list (unlikely but possible in bad GML)
        if isinstance(g_data, list):
            g_data = g_data[0]

        # Meta
        for k, v in g_data.items():
            if k == "directed":
                network.directed = bool(v)
            elif k not in ["node", "edge"]:
                network.meta[k] = v

        # Nodes
        nodes = g_data.get("node", [])
        if isinstance(nodes, dict):
            nodes = [nodes]  # Single node case

        for n in nodes:
            nid = str(n.get("id", ""))
            label = n.get("label", nid)
            node_obj = Node(nid, label)

            for k, v in n.items():
                if k in ["id", "label"]:
                    continue
                if k == "graphics":
                    # standard GML graphics
                    if "x" in v and "y" in v:
                        node_obj.viz["position"] = {
                            "x": v["x"],
                            "y": v["y"],
                            "z": v.get("z", 0),
                        }
                    if "fill" in v:
                        # GML colors are often hex strings or names
                        node_obj.viz["color_raw"] = v["fill"]
                else:
                    node_obj.attributes[k] = v

            network.add_node(node_obj)

        # Edges
        edges = g_data.get("edge", [])
        if isinstance(edges, dict):
            edges = [edges]

        for i, e in enumerate(edges):
            sid = str(e.get("source"))
            tid = str(e.get("target"))
            eid = str(e.get("id", i))
            weight = float(e.get("weight", 1.0))  # value is standard GML weight?
            if "value" in e:
                weight = float(e["value"])

            edge_obj = Edge(eid, sid, tid, weight)

            for k, v in e.items():
                if k in ["id", "source", "target", "weight", "value"]:
                    continue
                edge_obj.attributes[k] = v

            network.add_edge(edge_obj)

        return network

    def stringify(self, network):
        lines = ["graph ["]
        indent = "  "

        # Meta
        for k, v in network.meta.items():
            if isinstance(v, str):
                v = f'"{v}"'
            lines.append(f"{indent}{k} {v}")

        if network.directed:
            lines.append(f"{indent}directed 1")
        else:
            lines.append(f"{indent}directed 0")

        # Nodes
        for node in network.nodes.values():
            lines.append(f"{indent}node [")
            lines.append(f"{indent}  id {node.id}")
            label = node.label.replace('"', '\\"')
            lines.append(f'{indent}  label "{label}"')

            # Attributes
            for k, v in node.attributes.items():
                if isinstance(v, str):
                    v = f'"{v}"'
                lines.append(f"{indent}  {k} {v}")

            # Viz (graphics)
            if "position" in node.viz:
                pos = node.viz["position"]
                lines.append(f"{indent}  graphics [")
                lines.append(f"{indent}    x {pos.get('x', 0)}")
                lines.append(f"{indent}    y {pos.get('y', 0)}")
                if "z" in pos:
                    lines.append(f"{indent}    z {pos.get('z', 0)}")
                lines.append(f"{indent}  ]")

            lines.append(f"{indent}]")

        # Edges
        for edge in network.edges:
            lines.append(f"{indent}edge [")
            lines.append(f"{indent}  source {edge.source}")
            lines.append(f"{indent}  target {edge.target}")
            lines.append(f"{indent}  weight {edge.weight}")

            for k, v in edge.attributes.items():
                if isinstance(v, str):
                    v = f'"{v}"'
                lines.append(f"{indent}  {k} {v}")

            lines.append(f"{indent}]")

        lines.append("]")
        return "\n".join(lines)


class XNETParser(BaseParser):
    """Parses and Writes XNET format (Line-based format)."""

    def parse(self, data):
        lines = data.split("\n")
        status = {"index": 0, "lines": lines}
        network = Network()

        # 1. Read Vertices Header
        while status["index"] < len(lines) and not lines[status["index"]].strip():
            status["index"] += 1

        header = lines[status["index"]].split()
        if not header or header[0].lower() != "#vertices":
            raise ValueError("Malformed XNET: Missing #vertices")

        # node_count = int(header[1]) # Unused, we rely on reading labels
        status["index"] += 1

        # 2. Read Labels (Nodes)
        labels = []
        while status["index"] < len(lines):
            line = lines[status["index"]].strip()
            if not line:
                status["index"] += 1
                continue
            if line.startswith("#"):
                break  # Next section

            label = line
            if label.startswith('"') and label.endswith('"'):
                label = label[1:-1]
            labels.append(label)
            status["index"] += 1

        # Create Nodes. XNET implicitly indexes nodes 0..N
        for i, label in enumerate(labels):
            network.add_node(Node(str(i), label))

        # 3. Read Edges Header
        while status["index"] < len(lines) and not lines[status["index"]].strip():
            status["index"] += 1

        edge_header_line = lines[status["index"]]
        edge_header = edge_header_line.split()
        if not edge_header or edge_header[0].lower() != "#edges":
            raise ValueError("Malformed XNET: Missing #edges")

        network.mode = "static"  # XNET is usually static
        is_weighted = (
            "weighted" in edge_header_line.lower()
            and "nonweighted" not in edge_header_line.lower()
        )
        is_directed = (
            "directed" in edge_header_line.lower()
            and "undirected" not in edge_header_line.lower()
        )
        network.directed = is_directed
        status["index"] += 1

        # 4. Read Edges
        edge_idx = 0
        while status["index"] < len(lines):
            line = lines[status["index"]].strip()
            if not line:
                status["index"] += 1
                continue
            if line.startswith("#"):
                break

            parts = line.split()
            if len(parts) < 2:
                continue

            src = parts[0]
            tgt = parts[1]
            w = float(parts[2]) if len(parts) > 2 else 1.0

            network.add_edge(Edge(str(edge_idx), src, tgt, w, network.edge_type))
            edge_idx += 1
            status["index"] += 1

        # 5. Properties (Vertices and Edges)
        while status["index"] < len(lines):
            line = lines[status["index"]].strip()
            if not line:
                status["index"] += 1
                continue

            # Expect header: #v "Name" type
            match = re.match(r'#([ve]) "(.+)" ([sn]|v2|v3)', line)
            if not match:
                # unknown line or EOF
                status["index"] += 1
                continue

            prop_type = match.group(1)  # v or e
            prop_name = match.group(2)
            prop_fmt = match.group(3)

            status["index"] += 1

            # Read values
            values = []
            while status["index"] < len(lines):
                line = lines[status["index"]].strip()
                if not line:
                    status["index"] += 1
                    continue
                if line.startswith("#"):
                    break

                # Parse value based on format
                val = line
                if val.startswith('"') and val.endswith('"'):
                    val = val[1:-1]

                parsed_val = val
                if prop_fmt == "n":
                    try:
                        parsed_val = float(val)
                    except:
                        parsed_val = 0.0
                elif prop_fmt == "v2":
                    parts = val.split()
                    if len(parts) >= 2:
                        parsed_val = [float(parts[0]), float(parts[1])]
                elif prop_fmt == "v3":
                    parts = val.split()
                    if len(parts) >= 3:
                        parsed_val = [float(parts[0]), float(parts[1]), float(parts[2])]

                values.append(parsed_val)
                status["index"] += 1

            # Assign values
            if prop_type == "v":
                # Vertices properties (assume order matches ID order 0..N)
                # Note: XNET assumes nodes are 0..N indices
                for i, val in enumerate(values):
                    nid = str(i)
                    if nid in network.nodes:
                        if (
                            prop_name.lower() == "position"
                            and isinstance(val, list)
                            and len(val) >= 2
                        ):
                            network.nodes[nid].viz["position"] = {
                                "x": val[0],
                                "y": val[1],
                                "z": val[2] if len(val) > 2 else 0,
                            }
                        elif prop_name.lower() == "color":
                            # XNET colors might be v3 (rgb)
                            if isinstance(val, list):
                                network.nodes[nid].viz["color"] = (
                                    val  # Assumed normalized
                                )
                        else:
                            network.nodes[nid].attributes[prop_name] = val
            elif prop_type == "e":
                for i, val in enumerate(values):
                    if i < len(network.edges):
                        network.edges[i].attributes[prop_name] = val

        return network

    def stringify(self, network):
        lines = []

        # We need integer indices for XNET. Map Node ID -> Index
        node_id_to_idx = {nid: i for i, nid in enumerate(network.nodes.keys())}
        sorted_nodes = list(network.nodes.values())  # Order matches index 0..N

        # #vertices N
        lines.append(f"#vertices {len(sorted_nodes)}")
        for n in sorted_nodes:
            label = n.label.replace('"', '\\"')
            lines.append(f'"{label}"')

        # #edges weighted directed
        w_str = (
            "weighted" if any(e.weight != 1.0 for e in network.edges) else "nonweighted"
        )
        d_str = network.edge_type
        lines.append(f"#edges {w_str} {d_str}")

        for e in network.edges:
            s_idx = node_id_to_idx.get(e.source)
            t_idx = node_id_to_idx.get(e.target)
            if s_idx is not None and t_idx is not None:
                if w_str == "weighted":
                    lines.append(f"{s_idx} {t_idx} {e.weight}")
                else:
                    lines.append(f"{s_idx} {t_idx}")

        # Properties
        # We collect all properties from nodes/edges to see what columns to create
        # This is a simplification. Ideally check the model.

        # Vertex Properties
        v_props = {}  # key -> list of values (aligned with node index)
        for i, n in enumerate(sorted_nodes):
            for k, v in n.attributes.items():
                if k not in v_props:
                    v_props[k] = [None] * len(sorted_nodes)
                v_props[k][i] = v

        for key, values in v_props.items():
            # Determine type
            is_num = all(isinstance(x, (int, float)) or x is None for x in values)
            fmt = "n" if is_num else "s"
            lines.append(f'#v "{key}" {fmt}')
            for v in values:
                val_str = str(v) if v is not None else "0" if fmt == "n" else ""
                if fmt == "s":
                    val_str = f'"{val_str}"'
                lines.append(val_str)

        return "\n".join(lines)


_parsers = {"gexf": GEXFParser(), "gml": GMLParser(), "xnet": XNETParser()}


def parse_network(data, format):
    """
    Parses string data into a Network object.

    Args:
        data: The raw string content of the file.
        format: 'gexf', 'gml', or 'xnet'.
    """
    fmt = format.lower().strip()
    if fmt not in _parsers:
        raise ValueError(
            f"Unsupported format: {fmt}. Supported: {list(_parsers.keys())}"
        )

    return _parsers[fmt].parse(data)


def stringify_network(network, format):
    """
    Converts a Network object into a string of the specified format.

    Args:
        network: The Network object.
        format: 'gexf', 'gml', or 'xnet'.
    """
    fmt = format.lower().strip()
    if fmt not in _parsers:
        raise ValueError(
            f"Unsupported format: {fmt}. Supported: {list(_parsers.keys())}"
        )

    return _parsers[fmt].stringify(network)
