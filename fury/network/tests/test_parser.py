import numpy.testing as npt
import pytest

from fury.network.core import Edge, Network, Node
from fury.network.parser import (
    GEXFParser,
    GMLParser,
    XNETParser,
    parse_network,
    stringify_network,
)

# --- Fixtures ---


@pytest.fixture
def sample_gexf_data():
    return """<?xml version="1.0" encoding="UTF-8"?>
<gexf xmlns="http://www.gexf.net/1.2draft" xmlns:viz="http://www.gexf.net/1.2draft/viz" version="1.2">
    <meta>
        <creator>Fury</creator>
    </meta>
    <graph mode="static" defaultedgetype="directed">
        <attributes class="node">
            <attribute id="0" title="score" type="float"/>
            <attribute id="1" title="valid" type="boolean"/>
        </attributes>
        <nodes>
            <node id="n1" label="Node 1">
                <attvalues>
                    <attvalue for="0" value="0.5"/>
                    <attvalue for="1" value="true"/>
                </attvalues>
                <viz:color r="255" g="0" b="0"/>
                <viz:position x="10.0" y="20.0" z="5.0"/>
                <viz:size value="2.0"/>
            </node>
            <node id="n2" label="Node 2"/>
        </nodes>
        <edges>
            <edge id="e1" source="n1" target="n2" weight="2.5" type="directed"/>
        </edges>
    </graph>
</gexf>"""


@pytest.fixture
def sample_gml_data():
    return """graph [
  directed 1
  comment "This is a sample graph"
  node [
    id 1
    label "Node 1"
    score 0.99
    graphics [
        x 15.0
        y 25.0
        z 0.0
        fill "#00FF00"
    ]
  ]
  node [
    id 2
    label "Node 2"
  ]
  edge [
    id "e1"
    source 1
    target 2
    weight 1.5
    label "connection"
  ]
]"""


@pytest.fixture
def sample_xnet_data():
    return """#vertices 2
"Node A"
"Node B"
#edges weighted directed
0 1 5.5
#v "score" n
10.0
20.0
#v "flags" s
"active"
"inactive"
"""


# --- GEXF Parser Tests ---


def test_gexf_parse_valid(sample_gexf_data):
    network = parse_network(sample_gexf_data, "gexf")

    assert len(network.nodes) == 2
    assert len(network.edges) == 1
    assert network.edge_type == "directed"
    assert network.meta["creator"] == "Fury"

    # Check Node 1
    n1 = network.nodes["n1"]
    assert n1.label == "Node 1"
    assert n1.attributes["score"] == 0.5
    assert n1.attributes["valid"] is True

    # Check Viz
    npt.assert_array_equal(n1.viz["color"], [1.0, 0.0, 0.0])
    assert n1.viz["position"] == {"x": 10.0, "y": 20.0, "z": 5.0}
    assert n1.viz["size"] == 2.0

    # Check Edge
    e1 = network.edges[0]
    assert e1.source == "n1"
    assert e1.target == "n2"
    assert e1.weight == 2.5


def test_gexf_parse_invalid_xml():
    invalid_xml = "<gexf><unclosed_tag>"
    with pytest.raises(ValueError, match="Invalid GEXF XML Data"):
        parse_network(invalid_xml, "gexf")


def test_gexf_parse_missing_graph_tag():
    xml = "<gexf><meta></meta></gexf>"
    with pytest.raises(ValueError, match="No <graph> tag found"):
        parse_network(xml, "gexf")


def test_gexf_roundtrip(sample_gexf_data):
    # Parse -> Stringify -> Parse -> Compare
    original_net = parse_network(sample_gexf_data, "gexf")
    gexf_str = stringify_network(original_net, "gexf")
    final_net = parse_network(gexf_str, "gexf")

    assert len(final_net.nodes) == len(original_net.nodes)
    assert len(final_net.edges) == len(original_net.edges)

    # Check if attributes persisted
    n1 = final_net.nodes["n1"]
    assert n1.attributes["score"] == 0.5
    # Note: Attributes might come back as strings if not strictly typed in roundtrip without schema persistence
    # But our parser handles type mapping if model exists.


# --- GML Parser Tests ---


def test_gml_parse_valid(sample_gml_data):
    network = parse_network(sample_gml_data, "gml")

    assert len(network.nodes) == 2
    assert network.directed is True  # from "directed 1"

    # Check Node 1 (ID is string "1")
    n1 = network.nodes["1"]
    assert n1.label == "Node 1"
    assert n1.attributes["score"] == 0.99

    # Check Graphics
    assert n1.viz["position"]["x"] == 15.0
    assert n1.viz["color_raw"] == "#00FF00"

    # Check Edge
    e1 = network.edges[0]
    assert e1.source == "1"
    assert e1.target == "2"
    assert e1.weight == 1.5


def test_gml_parse_missing_graph():
    gml = "node [ id 1 ]"
    with pytest.raises(ValueError, match="GML must contain a 'graph' key"):
        parse_network(gml, "gml")


def test_gml_parse_duplicate_keys():
    # GML often has multiple 'node' keys, our parser must handle this as list
    gml = """graph [
        node [ id 1 ]
        node [ id 2 ]
    ]"""
    network = parse_network(gml, "gml")
    assert len(network.nodes) == 2


def test_gml_roundtrip(sample_gml_data):
    original_net = parse_network(sample_gml_data, "gml")
    gml_str = stringify_network(original_net, "gml")
    final_net = parse_network(gml_str, "gml")

    assert len(final_net.nodes) == len(original_net.nodes)
    assert final_net.nodes["1"].label == "Node 1"
    assert final_net.nodes["1"].viz["position"]["x"] == 15.0


# --- XNET Parser Tests ---


def test_xnet_parse_valid(sample_xnet_data):
    network = parse_network(sample_xnet_data, "xnet")

    # XNET assigns IDs 0, 1, 2...
    assert len(network.nodes) == 2
    assert network.edge_type == "directed"

    n0 = network.nodes["0"]
    assert n0.label == "Node A"
    assert n0.attributes["score"] == 10.0
    assert n0.attributes["flags"] == "active"

    n1 = network.nodes["1"]
    assert n1.label == "Node B"
    assert n1.attributes["score"] == 20.0

    # Edge 0->1
    assert len(network.edges) == 1
    e0 = network.edges[0]
    assert e0.source == "0"
    assert e0.target == "1"
    assert e0.weight == 5.5


def test_xnet_parse_malformed_headers():
    data = "#invalid_header\n..."
    with pytest.raises(ValueError, match="Malformed XNET"):
        parse_network(data, "xnet")

    data_no_edges = '#vertices 1\n"A"\n#bad_edge_header'
    with pytest.raises(ValueError, match="Malformed XNET"):
        parse_network(data_no_edges, "xnet")


def test_xnet_roundtrip(sample_xnet_data):
    original_net = parse_network(sample_xnet_data, "xnet")
    xnet_str = stringify_network(original_net, "xnet")
    final_net = parse_network(xnet_str, "xnet")

    assert len(final_net.nodes) == 2
    assert final_net.nodes["0"].label == "Node A"
    assert final_net.nodes["0"].attributes["score"] == 10.0


# --- General API & Edge Cases ---


def test_parse_network_invalid_format():
    with pytest.raises(ValueError, match="Unsupported format"):
        parse_network("data", "invalid_fmt")


def test_stringify_network_invalid_format():
    net = Network()
    with pytest.raises(ValueError, match="Unsupported format"):
        stringify_network(net, "invalid_fmt")


def test_empty_network_roundtrip():
    net = Network()
    # Should work for all formats even if empty
    for fmt in ["gexf", "gml", "xnet"]:
        s = stringify_network(net, fmt)
        res = parse_network(s, fmt)
        assert len(res.nodes) == 0
        assert len(res.edges) == 0


def test_network_construction_manual():
    # Verify manual construction used in stringify
    net = Network()
    n1 = Node("1", "A")
    n1.attributes["val"] = 10
    n1.viz["color"] = [0.0, 1.0, 0.0]

    n2 = Node("2", "B")

    e1 = Edge("e1", "1", "2", weight=0.5)

    net.add_node(n1)
    net.add_node(n2)
    net.add_edge(e1)

    # Quick GML stringify check
    gml = stringify_network(net, "gml")
    assert 'label "A"' in gml
