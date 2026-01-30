"""
=======================================
Force-Directed Network Visualization
=======================================

This example demonstrates how to use the `Network` actor to visualize
graphs using a GPU-accelerated Fruchterman-Reingold force-directed layout.

It supports two modes:
1. Dummy: Generates a random synthetic graph.
2. File: Loads a graph from .gexf, .gml, or .xnet files.

"""

import argparse
import os
import numpy as np
from fury import window, ui
from fury.actor.network import Network as NetworkActor
from fury.network.parser import parse_network


def get_random_colors(n, alpha=1.0):
    """Generate random RGBA colors."""
    colors = np.random.rand(n, 4).astype(np.float32)
    colors[:, 3] = alpha
    return colors


def create_dummy_data(n_nodes=100, n_edges=200):
    """Generate random nodes and edges."""
    print(f"Generating dummy graph with {n_nodes} nodes and {n_edges} edges...")

    # Random positions inside a cube
    nodes = (np.random.rand(n_nodes, 3) - 0.5) * 100

    # Random edges
    edges = np.random.randint(0, n_nodes, size=(n_edges, 2))
    # Remove self-loops
    edges = edges[edges[:, 0] != edges[:, 1]]

    colors = get_random_colors(n_nodes)

    return nodes, edges, colors


def adapter_core_to_actor(core_network):
    """Convert a fury.network.core.Network object to Actor input arrays.

    Parameters
    ----------
    core_network : fury.network.core.Network

    Returns
    -------
    tuple
        (nodes_xyz, edges_indices, colors)
    """
    node_ids = list(core_network.nodes.keys())
    id_to_idx = {id_: i for i, id_ in enumerate(node_ids)}
    n_nodes = len(node_ids)

    # 1. Extract Positions
    # If positions are missing in the file, initialize randomly
    nodes_xyz = np.zeros((n_nodes, 3), dtype=np.float32)
    has_layout = False

    for i, node_id in enumerate(node_ids):
        node = core_network.nodes[node_id]
        pos = node.viz.get("position")
        if pos:
            has_layout = True
            nodes_xyz[i] = [pos.get("x", 0), pos.get("y", 0), pos.get("z", 0)]
        else:
            # Random initialization if no layout
            nodes_xyz[i] = (np.random.rand(3) - 0.5) * 50

    # If the layout seems completely flat or empty, maybe add some jitter
    if not has_layout or np.all(nodes_xyz == 0):
        print("No layout found in file. Initializing random positions.")
        nodes_xyz = (np.random.rand(n_nodes, 3) - 0.5) * 100

    # 2. Extract Edges
    edges_list = []
    for edge in core_network.edges:
        u = id_to_idx.get(edge.source)
        v = id_to_idx.get(edge.target)

        # Only add valid edges where both nodes exist
        if u is not None and v is not None and u != v:
            edges_list.append([u, v])

    edges_indices = np.array(edges_list, dtype=np.int32)
    if len(edges_indices) == 0:
        edges_indices = np.zeros((0, 2), dtype=np.int32)

    # 3. Extract Colors
    colors = np.ones((n_nodes, 4), dtype=np.float32)
    for i, node_id in enumerate(node_ids):
        node = core_network.nodes[node_id]
        c = node.viz.get("color")
        if c:
            # viz['color'] is typically [r, g, b] 0-1 float
            if len(c) >= 3:
                colors[i, :3] = c[:3]
                if len(c) > 3:
                    colors[i, 3] = c[3]

    return nodes_xyz, edges_indices, colors


def load_file_data(filepath):
    """Load and parse network file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    _, ext = os.path.splitext(filepath)
    fmt = ext.lower().replace(".", "")

    valid_formats = ["gexf", "gml", "xnet"]
    if fmt not in valid_formats:
        raise ValueError(f"Unsupported format '{fmt}'. Must be one of {valid_formats}")

    print(f"Loading {filepath} as {fmt}...")

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    core_net = parse_network(content, fmt)
    print(f"Parsed {len(core_net.nodes)} nodes and {len(core_net.edges)} edges.")

    return adapter_core_to_actor(core_net)


def main():
    parser = argparse.ArgumentParser(description="FURY Network Visualization Demo")
    parser.add_argument(
        "--file", type=str, help="Path to a graph file (.gexf, .gml, .xnet)"
    )
    parser.add_argument(
        "--nodes", type=int, default=500, help="Number of nodes for dummy graph"
    )
    parser.add_argument(
        "--edges", type=int, default=1000, help="Number of edges for dummy graph"
    )
    parser.add_argument("--k", type=float, default=20.0, help="Optimal distance (K)")
    parser.add_argument("--speed", type=float, default=0.01, help="Simulation speed")

    args = parser.parse_args()

    # 1. Get Data
    if args.file:
        try:
            nodes, edges, colors = load_file_data(args.file)
        except Exception as e:
            print(f"Error loading file: {e}")
            return
    else:
        nodes, edges, colors = create_dummy_data(args.nodes, args.edges)

    # 2. Create Actor
    # Using the Network actor from fury/actor/network.py
    network_actor = NetworkActor(
        nodes=nodes,
        edges=edges,
        colors=colors,
        k=args.k,
        damping=0.9,
        repulsion_strength=1.0,
        speed=args.speed,
    )
    # network_actor.local.position = (10, 1, 10)

    # 3. Visualization Setup
    scene = window.Scene()
    scene.background = (0.1, 0.1, 0.1)
    scene.add(network_actor)

    # Add some UI info
    # info_text = f"Nodes: {len(nodes)}\nEdges: {len(edges)}\nMode: {'File' if args.file else 'Dummy'}"
    # ui_label = ui.TextBlock2D(
    #     info_text, position=(10, 10), font_size=16, color=(1, 1, 1)
    # )
    # scene.add(ui_label)

    show_m = window.ShowManager(
        scene=scene,
        size=(1024, 768),
        title="FURY Network - Compute Shader Force Layout",
    )

    # # 4. Interaction
    # # Allow adjusting simulation parameters at runtime
    # def timer_callback():
    #     # We can update uniforms dynamically if needed,
    #     # e.g. decay temperature or handle user inputs.
    #     # For now, the shader runs continuously.
    #     # network_actor.material.uniform_buffer.update_full()
    #     # show_m.render()
    #     print(network_actor.local.position)
    #     print(show_m.screens[0].camera.local.position)

    # # Register the callback (True for repeat, 16ms duration)
    # show_m.register_callback(timer_callback, 1, True, "Network Update")

    # Ensure camera is updated before start
    # window.update_camera(show_m.screens[0].camera, None, scene)

    print("\nStarting simulation... Close window to exit.")
    show_m.start()


if __name__ == "__main__":
    main()
