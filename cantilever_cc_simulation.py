import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy.stats import cauchy
import seaborn as sns
from matplotlib.animation import FuncAnimation
from typing import List, Tuple

@dataclass
class ColorCenter:
    """Represents a color center in the cantilever"""
    id: int
    base_frequency: float  # Base transition frequency at zero strain (Hz)
    strain_coefficient: float  # Hz/V of frequency shift due to strain
    position_z: float  # Normalized position in beam (-1 to 1, 0 is center)
    linewidth: float  # Lorentzian linewidth (Hz)
    
    def __post_init__(self):
        """Validate parameters after initialization"""
        assert -1 <= self.position_z <= 1, f"Position must be in [-1,1], got {self.position_z}"
        assert self.base_frequency > 0, "Base frequency must be positive"
        assert self.linewidth > 0, "Linewidth must be positive"
    
    def frequency_at_displacement(self, s: float) -> float:
        """Calculate frequency at given displacement"""
        new_freq = self.base_frequency + self.strain_coefficient * s
        # Return minimum frequency of linewidth/2 to ensure always positive
        return max(new_freq, self.linewidth/2)

@dataclass
class SimulationConfig:
    """Configuration parameters for the cantilever simulation"""
    n_color_centers: int = 40
    center_frequency: float = 400e12  # 400 THz
    frequency_spread: float = 100e9   # 100 GHz FWHM
    linewidth: float = 200e6          # 200 MHz
    max_strain_tuning: float = 20e9   # 20 GHz
    random_seed: int = 42
    max_displacement: float = 1.0     # Maximum cantilever displacement
    entanglement_threshold: float = 200e6  # Frequency difference threshold for entanglement (200 MHz)
    p_entanglement: float = 0.1  # Probability of successful entanglement

    def validate(self) -> None:
        """Validate configuration parameters"""
        assert self.n_color_centers > 0, "Number of color centers must be positive"
        assert self.max_displacement <= 1.0, "Maximum displacement must be <= 1.0"
        assert self.linewidth > 0, "Linewidth must be positive"
        assert self.frequency_spread > 0, "Frequency spread must be positive"
        assert self.max_strain_tuning > 0, "Maximum strain tuning must be positive"

def create_color_centers(config: SimulationConfig) -> List[ColorCenter]:
    """Create random distribution of color centers"""
    np.random.seed(config.random_seed)
    
    # Generate random base frequencies (Gaussian distribution)
    base_frequencies = np.random.normal(
        loc=config.center_frequency,
        scale=config.frequency_spread/2.355,
        size=config.n_color_centers
    )
    
    # Generate random positions in the beam
    positions = np.random.uniform(-1, 1, config.n_color_centers)
    
    return [
        ColorCenter(
            id=i,
            base_frequency=base_frequencies[i],
            strain_coefficient=config.max_strain_tuning * abs(positions[i]),
            position_z=positions[i],
            linewidth=config.linewidth
        )
        for i in range(config.n_color_centers)
    ]

def plot_phase_space(color_centers: List[ColorCenter], config: SimulationConfig, 
                    displacement: float = 0.0, fig=None, ax=None):
    """Plot color centers in phase space (f vs df/ds) with current edges"""
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        ax.clear()
    
    # Convert to GHz for better readability
    base_freqs = [(cc.base_frequency - config.center_frequency)/1e9 for cc in color_centers]
    strain_coeffs = [cc.strain_coefficient/1e9 for cc in color_centers]
    
    # Plot each color center
    scatter = ax.scatter(base_freqs, strain_coeffs, s=100, alpha=0.6)
    
    # Add labels for each point
    for i, cc in enumerate(color_centers):
        ax.annotate(f'CC{cc.id}', 
                   (base_freqs[i], strain_coeffs[i]),
                   xytext=(5, 5), textcoords='offset points')
    
    # Find possible entanglement edges
    current_freqs = [cc.frequency_at_displacement(displacement) for cc in color_centers]
    edges = []
    for i, cc1 in enumerate(color_centers):
        for j, cc2 in enumerate(color_centers[i+1:], i+1):
            freq_diff = abs(current_freqs[i] - current_freqs[j])
            if freq_diff < config.entanglement_threshold:
                edges.append((i, j))
                # Draw edge
                ax.plot([base_freqs[i], base_freqs[j]], 
                       [strain_coeffs[i], strain_coeffs[j]], 
                       'r--', alpha=0.5)
    
    ax.set_xlabel('Detuning from 400 THz (GHz)')
    ax.set_ylabel('Strain Coefficient (GHz/displacement)')
    ax.set_title(f'Color Center Phase Space (displacement = {displacement:.2f})')
    ax.grid(True)
    
    # Add displacement indicator
    ax.text(0.02, 0.98, f'Displacement: {displacement:.2f}', 
            transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8))
    
    return fig, ax, edges

def animate_phase_space(color_centers: List[ColorCenter], config: SimulationConfig):
    """Create animation of phase space as cantilever oscillates"""
    fig = plt.gcf()  # Get current figure
    fig.set_size_inches(12, 12)  # Make figure larger
    
    # Create three subplots
    gs = plt.GridSpec(3, 1, height_ratios=[2, 1, 1])
    ax_phase = fig.add_subplot(gs[0])    # Phase space plot
    ax_motion = fig.add_subplot(gs[1])   # Cantilever motion plot
    ax_beam = fig.add_subplot(gs[2])     # Beam cross section
    
    # Setup cantilever motion plot
    time_points = np.linspace(0, 2*np.pi, 100)
    motion_line, = ax_motion.plot([], [], 'b-')
    ax_motion.set_xlim(0, 2*np.pi)
    ax_motion.set_ylim(-1.1, 1.1)
    ax_motion.set_xlabel('Time')
    ax_motion.set_ylabel('Displacement')
    ax_motion.grid(True)
    
    # Setup beam cross section plot
    beam_x = np.linspace(-1, 1, 100)
    ax_beam.set_xlim(-1.1, 1.1)
    ax_beam.set_ylim(-0.2, 0.2)
    ax_beam.set_xlabel('Position (z)')
    ax_beam.set_ylabel('y')
    ax_beam.grid(True)
    
    # Store persistent entanglement edges
    persistent_edges = set()
    
    # Convert to GHz for better readability (calculate once)
    base_freqs = [(cc.base_frequency - config.center_frequency)/1e9 for cc in color_centers]
    strain_coeffs = [cc.strain_coefficient/1e9 for cc in color_centers]
    
    def init():
        ax_phase.clear()
        motion_line.set_data([], [])
        ax_beam.clear()
        return []
    
    def update(frame):
        ax_phase.clear()
        ax_beam.clear()
        
        # Calculate current displacement
        displacement = config.max_displacement * np.sin(2 * np.pi * frame / 100)
        
        # Update motion plot
        motion_line.set_data(time_points[:frame+1], 
                           config.max_displacement * np.sin(2 * np.pi * time_points[:frame+1] / (2*np.pi)))
        
        # Update beam plot
        beam_y = 0.1 * displacement * np.sin(np.pi * beam_x)  # Simple beam mode shape
        ax_beam.plot(beam_x, beam_y, 'b-', linewidth=2)
        ax_beam.fill(beam_x, beam_y, alpha=0.2, color='blue')
        ax_beam.fill(beam_x, -beam_y, alpha=0.2, color='blue')
        
        # Plot color centers in beam
        for cc in color_centers:
            y_pos = 0.1 * displacement * np.sin(np.pi * cc.position_z)
            ax_beam.scatter(cc.position_z, y_pos, color='red', s=50)
            ax_beam.annotate(f'CC{cc.id}', (cc.position_z, y_pos), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Plot color centers in phase space
        scatter = ax_phase.scatter(base_freqs, strain_coeffs, s=100, alpha=0.6, label='Color Centers')
        
        # Add labels to phase space
        for i, cc in enumerate(color_centers):
            ax_phase.annotate(f'CC{cc.id}', 
                          (base_freqs[i], strain_coeffs[i]),
                          xytext=(5, 5), textcoords='offset points')
        
        # Plot persistent edges
        for i, j in persistent_edges:
            ax_phase.plot([base_freqs[i], base_freqs[j]], 
                       [strain_coeffs[i], strain_coeffs[j]], 
                       'k-', alpha=0.5)
        
        # Calculate current frequencies and find new entanglement edges
        current_freqs = [cc.frequency_at_displacement(displacement) for cc in color_centers]
        has_edges = False
        for i in range(len(color_centers)):
            for j in range(i+1, len(color_centers)):
                freq_diff = abs(current_freqs[i] - current_freqs[j])
                if freq_diff < config.entanglement_threshold:
                    # Draw temporary edge
                    ax_phase.plot([base_freqs[i], base_freqs[j]], 
                              [strain_coeffs[i], strain_coeffs[j]], 
                              'r--', alpha=0.5, label='Potential Entanglement' if not has_edges else "")
                    has_edges = True
                    
                    # Draw entanglement line in beam plot
                    cc1_y = 0.1 * displacement * np.sin(np.pi * color_centers[i].position_z)
                    cc2_y = 0.1 * displacement * np.sin(np.pi * color_centers[j].position_z)
                    ax_beam.plot([color_centers[i].position_z, color_centers[j].position_z],
                               [cc1_y, cc2_y], 'r--', alpha=0.3)
                    
                    # Attempt entanglement with probability p_ent
                    if (i, j) not in persistent_edges and np.random.random() < config.p_entanglement:
                        persistent_edges.add((i, j))
        
        # Draw persistent edges in beam plot
        for i, j in persistent_edges:
            cc1_y = 0.1 * displacement * np.sin(np.pi * color_centers[i].position_z)
            cc2_y = 0.1 * displacement * np.sin(np.pi * color_centers[j].position_z)
            ax_beam.plot([color_centers[i].position_z, color_centers[j].position_z],
                        [cc1_y, cc2_y], 'k-', alpha=0.5)
        
        ax_phase.set_xlabel('Detuning from 400 THz (GHz)')
        ax_phase.set_ylabel('Strain Coefficient (GHz/displacement)')
        ax_phase.set_title(f'Color Center Phase Space (displacement = {displacement:.2f})')
        ax_phase.grid(True)
        
        # Set consistent axis limits
        ax_phase.set_xlim(min(base_freqs) - 1, max(base_freqs) + 1)
        ax_phase.set_ylim(min(strain_coeffs) - 1, max(strain_coeffs) + 1)
        
        # Add legend if there are edges
        if has_edges or persistent_edges:
            ax_phase.legend()
        
        plt.tight_layout()
        return [ax_phase, motion_line]
    
    ani = FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=100,
        interval=100,
        blit=False,
        repeat=True
    )
    
    return ani

def main():
    # Create configuration
    config = SimulationConfig()
    
    # Generate color centers
    color_centers = create_color_centers(config)
    
    # Print color center information
    print("\nColor Center Information:")
    for cc in color_centers:
        detuning = (cc.base_frequency - config.center_frequency)/1e9
        print(f"CC{cc.id}: Position={cc.position_z:.2f}, "
              f"Detuning={detuning:.2f} GHz, "
              f"Max Tuning={cc.strain_coefficient/1e9:.2f} GHz")

    # Create figure and animation
    fig = plt.figure(figsize=(10, 8))
    anim = animate_phase_space(color_centers, config)
    
    # Keep reference to prevent garbage collection
    fig.anim = anim  
    
    # Show animation
    plt.show()

if __name__ == "__main__":
    main() 