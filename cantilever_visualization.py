import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
from cantilever_cc_simulation import ColorCenter, SimulationConfig, create_color_centers
import networkx as nx

def animate_phase_space_with_beam(color_centers, config):
    """Enhanced animation showing phase space, motion, and beam side view"""
    # Physical dimensions in microns
    LENGTH = 10.0  # Length = 10 μm
    WIDTH = 1.0    # Width = 1 μm
    THICKNESS = 0.3  # Thickness = 300 nm = 0.3 μm
    MAX_DEFLECTION = 1.0  # Max displacement = 1 μm
    
    fig = plt.figure(figsize=(15, 12))
    
    # Create three subplots (removed y-z cross section)
    gs = GridSpec(3, 1, height_ratios=[2, 1, 1])
    ax_phase = fig.add_subplot(gs[0])     # Phase space plot
    ax_motion = fig.add_subplot(gs[1])    # Cantilever motion plot
    ax_beam_xz = fig.add_subplot(gs[2])   # Beam side view (x-z)
    
    # Setup phase space plot
    ax_phase.set_xlabel(r'Frequency Detuning, $\Delta\nu = \nu - \nu_0$ (GHz)')
    ax_phase.set_ylabel(r'Strain Coefficient, $\frac{d\nu}{dz}$ (GHz/μm)')
    ax_phase.set_title(r'Color Center Phase Space')
    ax_phase.grid(True)
    
    # Setup motion plot
    time_points = np.linspace(0, 2*np.pi, 100)
    motion_line, = ax_motion.plot([], [], 'b-')
    ax_motion.set_xlim(0, 2*np.pi)
    ax_motion.set_ylim(-MAX_DEFLECTION, MAX_DEFLECTION)
    ax_motion.set_xlabel(r'Normalized Time, $\omega t/2\pi$')
    ax_motion.set_ylabel(r'Tip Displacement, $z(t)$ (μm)')
    ax_motion.set_title(r'Cantilever Motion')
    ax_motion.grid(True)
    
    # Setup beam side view (x-z)
    ax_beam_xz.set_xlim(0, LENGTH)
    ax_beam_xz.set_ylim(-1, 1)
    ax_beam_xz.set_xlabel(r'Position Along Cantilever, $x/L$')
    ax_beam_xz.set_ylabel(r'Normalized Displacement, $z(x)/z_\mathrm{max}$')
    ax_beam_xz.set_title(r'Cantilever Mode Shape')
    ax_beam_xz.grid(True)
    
    # Store persistent entanglement edges
    persistent_edges = set()
    
    # Convert to GHz for better readability (calculate once)
    base_freqs = [(cc.base_frequency - config.center_frequency)/1e9 for cc in color_centers]
    strain_coeffs = [cc.strain_coefficient/1e9 for cc in color_centers]
    
    # Generate random positions for color centers (fixed throughout animation)
    cc_positions = {cc.id: {
        'x': np.random.uniform(0, LENGTH),  # Position along length
        'y': np.random.uniform(-WIDTH/2, WIDTH/2),  # Position across width
        'z': np.random.uniform(-THICKNESS/2, THICKNESS/2)  # Position in thickness
    } for cc in color_centers}
    
    def init():
        motion_line.set_data([], [])
        return [motion_line]
    
    def update(frame):
        ax_phase.clear()
        ax_beam_xz.clear()
        
        # Re-establish plot properties after clear
        ax_phase.set_xlabel(r'Frequency Detuning, $\Delta\nu = \nu - \nu_0$ (GHz)')
        ax_phase.set_ylabel(r'Strain Coefficient, $\frac{d\nu}{dz}$ (GHz/μm)')
        ax_phase.grid(True)
        ax_beam_xz.set_xlim(0, LENGTH)
        ax_beam_xz.set_ylim(-1, 1)
        ax_beam_xz.set_xlabel(r'Position Along Cantilever, $x/L$')
        ax_beam_xz.set_ylabel(r'Normalized Displacement, $z(x)/z_\mathrm{max}$')
        ax_beam_xz.grid(True)
        
        displacement = MAX_DEFLECTION * np.sin(2 * np.pi * frame / 100)
        
        # Find largest connected component
        G = nx.Graph()
        G.add_nodes_from(range(len(color_centers)))
        G.add_edges_from(persistent_edges)
        components = list(nx.connected_components(G))
        largest_component = max(components, key=len) if components else set()
        
        # Initialize flags for first occurrences
        first_largest = True
        first_other = True
        
        # Update motion plot (unchanged)
        motion_line.set_data(time_points[:frame+1], 
                           MAX_DEFLECTION * np.sin(2 * np.pi * time_points[:frame+1] / (2*np.pi)))
        
        # Update x-z side view
        x_points = np.linspace(0, LENGTH, 100)
        # Use realistic cantilever mode shape
        z_points = displacement * (1 - np.cos(np.pi * x_points / (2*LENGTH)))
        ax_beam_xz.fill_between(x_points, z_points-THICKNESS/2, z_points+THICKNESS/2, 
                              color='blue', alpha=0.3)
        
        # Plot color centers in phase space with coloring
        for i, cc in enumerate(color_centers):
            color = 'red' if i in largest_component else 'black'
            # Fix label color matching - only label first of each type
            if i in largest_component and first_largest:
                label = 'Largest Connected Component'  # Red points
                first_largest = False
            elif i not in largest_component and first_other:
                label = 'Others'    # Black points
                first_other = False
            else:
                label = None
            
            ax_phase.scatter(base_freqs[i], strain_coeffs[i], s=100, alpha=0.6, 
                           color=color, label=label)
            ax_phase.annotate(f'CC{cc.id}', 
                          (base_freqs[i], strain_coeffs[i]),
                          xytext=(5, 5), textcoords='offset points')
        
        # Calculate current frequencies and find optimal laser frequency
        current_freqs = [cc.frequency_at_displacement(displacement) for cc in color_centers]
        current_detunings = [(f - config.center_frequency)/1e9 for f in current_freqs]
        
        # Find pairs of close frequencies
        closest_pair = None
        min_diff = float('inf')
        for i in range(len(color_centers)):
            for j in range(i+1, len(color_centers)):
                freq_diff = abs(current_detunings[i] - current_detunings[j])
                if freq_diff < min_diff:
                    min_diff = freq_diff
                    closest_pair = (i, j)
        
        # Set laser frequency to average of closest pair
        if closest_pair and min_diff < config.entanglement_threshold/1e9:
            i, j = closest_pair
            laser_detuning = (current_detunings[closest_pair[0]] + current_detunings[closest_pair[1]])/2
            ax_phase.axvline(x=laser_detuning, color='green', linestyle='-', alpha=0.5,
                           label=f'Laser Frequency ({laser_detuning:.1f} GHz)')
            ax_phase.fill_between([laser_detuning-config.entanglement_threshold/1e9,
                                 laser_detuning+config.entanglement_threshold/1e9],
                                ax_phase.get_ylim()[0], ax_phase.get_ylim()[1],
                                color='green', alpha=0.1)
            
            # Add text showing frequency difference between closest pair
            ax_phase.text(laser_detuning, ax_phase.get_ylim()[1], 
                        f'Δf = {min_diff:.2f} GHz', 
                        color='green', ha='center', va='bottom',
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
            
            # Connect closest pair with green dashed line in phase space
            ax_phase.plot([base_freqs[i], base_freqs[j]], 
                         [strain_coeffs[i], strain_coeffs[j]], 
                         color='green', linestyle='--', linewidth=2, alpha=0.8)
            
            # Connect in beam view too
            pos_i = cc_positions[color_centers[i].id]
            pos_j = cc_positions[color_centers[j].id]
            defl_i = displacement * (1 - np.cos(np.pi * pos_i['x'] / (2*LENGTH)))
            defl_j = displacement * (1 - np.cos(np.pi * pos_j['x'] / (2*LENGTH)))
            ax_beam_xz.plot([pos_i['x'], pos_j['x']], 
                          [defl_i + pos_i['z'], defl_j + pos_j['z']],
                          color='green', linestyle='--', linewidth=2, alpha=0.8)
        
        # Plot color centers in beam view
        for i, cc in enumerate(color_centers):
            pos = cc_positions[cc.id]
            # Calculate actual deflection at this x position
            local_deflection = displacement * (1 - np.cos(np.pi * pos['x'] / (2*LENGTH)))
            # Add CC's z position to the deflection
            z_pos = local_deflection + pos['z']
            color = 'red' if i in largest_component else 'black'
            
            ax_beam_xz.scatter(pos['x'], z_pos,
                             color=color, s=20)
            ax_beam_xz.annotate(f'{cc.id}', (pos['x'], z_pos),
                              xytext=(2, 2), textcoords='offset points', fontsize=6)
        
        # Plot persistent edges with coloring
        for i, j in persistent_edges:
            color = 'red' if i in largest_component and j in largest_component else 'black'
            alpha = 0.7 if i in largest_component and j in largest_component else 0.3
            
            # In phase space
            ax_phase.plot([base_freqs[i], base_freqs[j]], 
                       [strain_coeffs[i], strain_coeffs[j]], 
                       color=color, alpha=alpha, linewidth=1.0)
            
            # In beam view
            pos_i = cc_positions[color_centers[i].id]
            pos_j = cc_positions[color_centers[j].id]
            defl_i = displacement * (1 - np.cos(np.pi * pos_i['x'] / (2*LENGTH)))
            defl_j = displacement * (1 - np.cos(np.pi * pos_j['x'] / (2*LENGTH)))
            ax_beam_xz.plot([pos_i['x'], pos_j['x']], 
                        [defl_i + pos_i['z'], defl_j + pos_j['z']],
                        color=color, alpha=alpha, linewidth=1.0)
        
        # Calculate current frequencies and find new entanglement edges
        current_freqs = [cc.frequency_at_displacement(displacement) for cc in color_centers]
        has_edges = False
        for i in range(len(color_centers)):
            for j in range(i+1, len(color_centers)):
                freq_diff = abs(current_freqs[i] - current_freqs[j])
                if freq_diff < config.entanglement_threshold:
                    pos_i = cc_positions[color_centers[i].id]
                    pos_j = cc_positions[color_centers[j].id]
                    
                    # Draw temporary edge in phase space
                    ax_phase.plot([base_freqs[i], base_freqs[j]], 
                              [strain_coeffs[i], strain_coeffs[j]], 
                              'r--', alpha=0.5, label='Potential Entanglement' if not has_edges else "")
                    has_edges = True
                    
                    # Calculate deflections for beam views
                    defl_i = displacement * (1 - np.cos(np.pi * pos_i['x'] / (2*LENGTH)))
                    defl_j = displacement * (1 - np.cos(np.pi * pos_j['x'] / (2*LENGTH)))
                    
                    # Draw in beam view
                    ax_beam_xz.plot([pos_i['x'], pos_j['x']], 
                                [defl_i + pos_i['z'], defl_j + pos_j['z']],
                                'r--', alpha=0.3, linewidth=0.5)
                    
                    # Attempt entanglement with probability p_ent
                    if (i, j) not in persistent_edges and np.random.random() < config.p_entanglement:
                        persistent_edges.add((i, j))
        
        # Update phase space title with current displacement
        ax_phase.set_title(r'Color Center Phase Space ($z(t) = {:.2f}$ μm)' '\n'
                          r'Largest Connected Component: $N = {%d}$ centers' % len(largest_component))
        
        # Update motion plot title
        ax_motion.set_title(r'Cantilever Motion vs. Time')
        
        # Update beam view title
        ax_beam_xz.set_title(r'Cantilever Side View ($x$-$z$ plane)')
        
        # Update legend with correct labels - only if it doesn't exist
        if persistent_edges and not ax_phase.get_legend():
            handles, labels = ax_phase.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            legend_labels = []
            if 'Largest Connected Component' in by_label:
                legend_labels.append('Largest Connected Component')  # Red points
            if 'Others' in by_label:
                legend_labels.append('Others')  # Black points
            if 'Laser Frequency' in by_label:
                legend_labels.append('Laser Frequency')  # Green line
            ax_phase.legend([by_label[label] for label in legend_labels], legend_labels, 
                          loc='upper right')  # Fix position to upper right
        
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
    
    return fig, ani

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

    # Create and show enhanced visualization
    fig, anim = animate_phase_space_with_beam(color_centers, config)
    
    # Keep reference to prevent garbage collection
    fig.anim = anim
    
    # Show animation
    plt.show()

if __name__ == "__main__":
    main() 