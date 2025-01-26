# -*- coding: utf-8 -*-
"""
Quantum Percolation Visualization on a Square Lattice

Features:
- Modular design with configuration management
- Type hints and comprehensive documentation
- Progress tracking for long simulations
- Efficient memory usage with precomputed values
- Improved error handling and validation
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from IPython.display import Image
import io
import tempfile
from dataclasses import dataclass
from typing import Set, Dict, Tuple, Generator, Any
from tqdm.notebook import tqdm
import matplotlib.lines as mlines

@dataclass
class SimulationConfig:
    """Configuration parameters for quantum percolation simulation."""
    n: int = 20        # Size of the lattice (n x n)
    steps: int = 80    # Number of time steps
    p_trial: float = 0.1 # Probability of establishing entanglement
    p_error: float = 0.01 # Probability of error
    fps: int = 2       # Animation frames per second
    seed: int = 42     # Random seed for reproducibility

    def validate(self) -> None:
        """Validate configuration parameters."""
        assert 0.0 <= self.p_trial <= 1.0, "p_trial must be between 0 and 1."
        assert 0.0 <= self.p_error <= 1.0, "p_error must be between 0 and 1."
        assert self.n > 0, "n must be positive."
        assert self.steps > 0, "steps must be positive."
        assert self.fps > 0, "fps must be positive."

def create_empty_graph(n: int) -> nx.Graph:
    """
    Creates an empty n x n lattice graph with no edges.

    Parameters
    ----------
    n : int
        Number of nodes along each dimension.

    Returns
    -------
    G : networkx.Graph
        Graph with nodes placed in an n x n grid.
    """
    G = nx.Graph()
    for x in range(n):
        for y in range(n):
            G.add_node((x, y))
    return G

def attempt_add_edges(G: nx.Graph, step: int, n: int, p_trial: float, 
                     trial_array: np.ndarray) -> None:
    """
    Attempt to add edges between nodes according to the given probabilities and pattern.

    Parameters
    ----------
    G : networkx.Graph
        The current graph.
    step : int
        Current simulation step.
    n : int
        Lattice size.
    p_trial : float
        Probability of establishing entanglement.
    trial_array : np.ndarray
        Precomputed random array for edge trials: shape (n, n, steps).
    """
    mod_step = step % 4
    for x in range(n):
        for y in range(n - 1):
            current_p = trial_array[x, y, step]
            if current_p < p_trial:
                if mod_step == 0 and x % 2 == 0 and y % 2 == 0:
                    G.add_edge((x, y), (x, y + 1))
                elif mod_step == 1 and x % 2 == 0 and x < n - 1:
                    G.add_edge((x, y), (x + 1, y))
                elif mod_step == 2 and x % 2 == 1 and y % 2 == 1:
                    G.add_edge((x, y), (x, y + 1))
                elif mod_step == 3 and x % 2 == 1 and x < n - 1:
                    G.add_edge((x, y), (x + 1, y))

def introduce_errors(G: nx.Graph, largest_cc: Set, error_array: np.ndarray, 
                    step: int, p_error: float, error_edges: Set) -> None:
    """
    Introduce heralded entanglement/decoherence errors at each step.

    Parameters
    ----------
    G : networkx.Graph
        Current state of the graph.
    largest_cc : set
        Largest connected component at this step.
    error_array : np.ndarray
        Array of random values for error introduction.
    step : int
        Current simulation step.
    p_error : float
        Probability of heralded entanglement/decoherence error.
    error_edges : set
        Set to store edges that experience heralded errors.
    """
    for node in list(G.nodes):
        if error_array[node[0] % len(error_array), step] < p_error:
            neighbors = list(G.neighbors(node))
            for neighbor in neighbors:
                if node in largest_cc and neighbor in largest_cc:
                    error_edges.add(tuple(sorted((node, neighbor))))
                G.remove_edge(node, neighbor)

def draw_grid(step: int, G: nx.Graph, largest_cc: Set, n: int, 
             ax: plt.Axes, error_edges: Set, pos: Dict[Tuple[int, int], Tuple[float, float]]) -> None:
    """
    Draw the square lattice graph with color-coded nodes and edges.

    Parameters
    ----------
    step : int
        Current simulation step.
    G : networkx.Graph
        Current state of the graph.
    largest_cc : set
        Largest connected component in the graph.
    n : int
        Lattice size.
    ax : matplotlib.axes.Axes
        Axes object for plotting.
    error_edges : set
        Set of edges with heralded entanglement or decoherence errors.
    pos : dict
        Precomputed positions of nodes.
    """
    ax.clear()

    node_colors = ['red' if node in largest_cc else 'gray' for node in G.nodes]

    edge_colors = []
    edge_styles = []
    for u, v in G.edges:
        edge_tuple = tuple(sorted((u, v)))
        if edge_tuple in error_edges:
            edge_colors.append('red')
            edge_styles.append('dashed')
        elif u in largest_cc and v in largest_cc:
            edge_colors.append('red')
            edge_styles.append('solid')
        else:
            edge_colors.append('black')
            edge_styles.append('solid')

    nx.draw(G, pos, node_color=node_colors, edge_color=edge_colors, style=edge_styles,
            with_labels=False, node_size=80, font_weight='bold', ax=ax)

    # Add legend
    data_qubit_marker = mlines.Line2D([], [], color='grey', marker='o', markersize=10,
                                      label='Data Qubit', linestyle='None')
    data_qubit_largest_marker = mlines.Line2D([], [], color='red', marker='o', markersize=10,
                                              label='Data Qubit in Largest Cluster', linestyle='None')
    entanglement_line = mlines.Line2D([], [], color='black', linestyle='-', linewidth=2,
                                      label='Entanglement')
    entangled_cluster_line = mlines.Line2D([], [], color='red', linestyle='-', linewidth=2,
                                           label='Largest Entangled Cluster')
    heralded_error_line = mlines.Line2D([], [], color='red', linestyle='dashed', linewidth=2,
                                        label='Heralded Error')

    ax.legend(handles=[data_qubit_marker, data_qubit_largest_marker, entanglement_line,
                       entangled_cluster_line, heralded_error_line], loc='upper right')

    ax.set_title(f"Time step: {step+1}")
    
    # Set proper axis settings
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-1, n)
    ax.set_ylim(-1, n)
    ax.autoscale(False)

def percolation_simulation(config: SimulationConfig) -> Generator[Tuple[int, nx.Graph, Set, Set], None, None]:
    """
    Simulates quantum percolation on a square lattice over a specified number of steps.
    
    Parameters
    ----------
    config : SimulationConfig
        Configuration parameters for the simulation.
    
    Yields
    ------
    Tuple[int, nx.Graph, Set, Set]
        Current step, graph state, largest connected component, and error edges.
    """
    config.validate()
    
    print(f"Starting simulation with {config.n}x{config.n} lattice for {config.steps} steps")
    print(f"Trial probability: {config.p_trial}, Error probability: {config.p_error}")
    
    np.random.seed(config.seed)
    
    G = create_empty_graph(config.n)
    error_edges = set()
    largest_cc = set()
    
    trial_array = np.random.rand(config.n, config.n, config.steps)
    error_array = np.random.rand(config.n, config.steps)
    
    for step in tqdm(range(config.steps), desc="Simulating"):
        attempt_add_edges(G, step, config.n, config.p_trial, trial_array)
        
        connected_components = nx.connected_components(G)
        largest_cc = max(connected_components, key=len) if G.edges else set()
        
        introduce_errors(G, largest_cc, error_array, step, config.p_error, error_edges)
        
        if G.edges:
            connected_components = nx.connected_components(G)
            largest_cc = max(connected_components, key=len)
        else:
            largest_cc = set()
            
        yield step, G, largest_cc, error_edges

def create_animation(config: SimulationConfig) -> Tuple[FuncAnimation, Dict]:
    """
    Create animation of the quantum percolation simulation.
    
    Parameters
    ----------
    config : SimulationConfig
        Configuration parameters for the simulation.
        
    Returns
    -------
    Tuple[FuncAnimation, Dict]
        Animation object and position dictionary for the nodes.
    """
    pos = {(x, y): (x, y) for x in range(config.n) for y in range(config.n)}
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    def update(frame):
        step, G, largest_cc, error_edges = frame
        draw_grid(step, G, largest_cc, config.n, ax, error_edges, pos)
    
    ani = FuncAnimation(
        fig, 
        update,
        frames=percolation_simulation(config),
        blit=False,
        interval=1000//config.fps,
        repeat=False,
        save_count=config.steps
    )
    
    plt.close(fig)
    return ani, pos

def save_animation(animation: FuncAnimation, config: SimulationConfig) -> Image:
    """
    Save the animation as a GIF and return it as an IPython Image.
    
    Parameters
    ----------
    animation : FuncAnimation
        The animation to save
    config : SimulationConfig
        Configuration parameters for the simulation
        
    Returns
    -------
    Image
        IPython Image object containing the animation
    """
    gif_writer = PillowWriter(fps=config.fps)
    
    with tempfile.NamedTemporaryFile(suffix=".gif") as tmpfile:
        animation.save(tmpfile.name, writer=gif_writer)
        with open(tmpfile.name, 'rb') as f:
            buf = io.BytesIO(f.read())
    
    return Image(buf.getvalue())

if __name__ == "__main__":
    # Create configuration
    config = SimulationConfig()
    
    # Create and save animation
    animation, pos = create_animation(config)
    
    # Save animation to a file
    gif_writer = PillowWriter(fps=config.fps)
    animation.save('quantum_percolation.gif', writer=gif_writer)
    
    print("Animation saved as 'quantum_percolation.gif'")
    
    # If running in notebook, also display it
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            display_image = save_animation(animation, config)
            display_image
    except ImportError:
        pass 