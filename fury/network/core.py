from fury.actor import Mesh


class Node:
    def __init__(self, id, label=""):
        self.id = str(id)
        self.label = label if label else str(id)
        self.attributes = {}
        # Viz: position={'x':0, 'y':0, 'z':0}, color=[r,g,b], size=1.0, shape='disc'
        self.viz = {}

    def to_dict(self):
        return {
            "id": self.id,
            "label": self.label,
            "attributes": self.attributes,
            "viz": self.viz,
        }


class Edge:
    def __init__(
        self,
        id,
        source,
        target,
        weight=1.0,
        type="undirected",
    ):
        self.id = str(id)
        self.source = str(source)
        self.target = str(target)
        self.weight = weight
        self.type = type
        self.attributes = {}
        self.viz = {}

    def to_dict(self):
        return {
            "id": self.id,
            "source": self.source,
            "target": self.target,
            "weight": self.weight,
            "type": self.type,
            "attributes": self.attributes,
            "viz": self.viz,
        }


class Network(Mesh):
    def __init__(self, directed=False):
        self.nodes = {}
        self.edges = []
        self.meta = {}
        self.mode = "static"
        self.directed = directed
        # Schema definitions for typed formats (GEXF).
        # Format: {'node': {'attr_title': 'type'}, 'edge': {'attr_title': 'type'}}
        self.model = {"node": {}, "edge": {}}

    @property
    def directed(self):
        return self.edge_type == "directed"

    @directed.setter
    def directed(self, value):
        self.edge_type = "directed" if value else "undirected"

    def add_node(self, node):
        self.nodes[node.id] = node

    def add_edge(self, edge):
        self.edges.append(edge)

    def to_dict(self):
        return {
            "meta": self.meta,
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "edges": [e.to_dict() for e in self.edges],
        }
