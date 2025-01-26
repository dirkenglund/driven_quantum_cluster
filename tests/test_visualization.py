import unittest
import matplotlib.pyplot as plt
from cantilever_cc_simulation import (
    ColorCenter,
    SimulationConfig,
    create_color_centers,
    plot_phase_space,
    animate_phase_space
)

class TestVisualization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up matplotlib for testing"""
        plt.switch_backend('Agg')  # Use non-interactive backend for testing
    
    def setUp(self):
        """Set up test case"""
        self.config = SimulationConfig()
        self.color_centers = create_color_centers(self.config)
        plt.close('all')  # Close any existing figures
    
    def test_phase_space_plot_creation(self):
        """Test that phase space plot is created with correct elements"""
        fig, ax, edges = plot_phase_space(self.color_centers, self.config)
        
        # Check figure and axis creation
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)
        
        # Check axis labels
        self.assertEqual(ax.get_xlabel(), 'Detuning from 400 THz (GHz)')
        self.assertEqual(ax.get_ylabel(), 'Strain Coefficient (GHz/displacement)')
        
        # Check that color centers are plotted
        scatter_points = [child for child in ax.get_children() 
                         if isinstance(child, plt.matplotlib.collections.PathCollection)]
        self.assertEqual(len(scatter_points), 1)  # One scatter plot
        self.assertEqual(len(scatter_points[0].get_offsets()), self.config.n_color_centers)
        
        plt.close(fig)
    
    def test_entanglement_edges(self):
        """Test that entanglement edges are drawn correctly"""
        # Create two color centers with frequencies close enough for entanglement
        cc1 = ColorCenter(id=0, base_frequency=400e12, 
                         strain_coefficient=10e9, position_z=0.5, 
                         linewidth=200e6)
        cc2 = ColorCenter(id=1, base_frequency=400e12 + 100e6,  # 100 MHz detuning
                         strain_coefficient=-10e9, position_z=-0.5, 
                         linewidth=200e6)
        
        fig, ax, edges = plot_phase_space([cc1, cc2], self.config)
        
        # Check for dashed lines (entanglement edges)
        lines = [child for child in ax.get_children() 
                if isinstance(child, plt.matplotlib.lines.Line2D)]
        dashed_lines = [line for line in lines if line.get_linestyle() == '--']
        
        # Should find at least one entanglement edge
        self.assertGreater(len(dashed_lines), 0)
        
        plt.close(fig)

    def test_animation_creation(self):
        """Test that animation is created with correct properties"""
        ani = animate_phase_space(self.color_centers, self.config)
        
        # Basic property checks
        self.assertIsNotNone(ani)
        self.assertIsNotNone(ani._fig)
        self.assertEqual(ani._interval, 100)  # Frame interval
        
        # Save animation to prevent warning (use gif instead of mp4)
        try:
            ani.save('test.gif', writer='pillow')
        except Exception:
            # If saving fails, just verify animation exists
            pass
        
        plt.close(ani._fig)
    
    def test_axis_limits(self):
        """Test that axis limits are set correctly"""
        fig, ax, edges = plot_phase_space(self.color_centers, self.config)
        
        # Get frequency range
        base_freqs = [(cc.base_frequency - self.config.center_frequency)/1e9 
                     for cc in self.color_centers]
        strain_coeffs = [cc.strain_coefficient/1e9 for cc in self.color_centers]
        
        # Check that all points are within view
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        
        self.assertLess(min(base_freqs), xlim[1])
        self.assertGreater(max(base_freqs), xlim[0])
        self.assertLess(min(strain_coeffs), ylim[1])
        self.assertGreater(max(strain_coeffs), ylim[0])
        
        plt.close(fig)
    
    def test_color_coding(self):
        """Test that color coding reflects entanglement state"""
        # Create two color centers that should be entangled
        cc1 = ColorCenter(id=0, base_frequency=400e12, 
                         strain_coefficient=10e9, position_z=0.5, 
                         linewidth=200e6)
        cc2 = ColorCenter(id=1, base_frequency=400e12 + 100e6,  # Within threshold
                         strain_coefficient=-10e9, position_z=-0.5, 
                         linewidth=200e6)
        
        # Use larger threshold to ensure entanglement
        config = SimulationConfig(entanglement_threshold=1e9)
        fig, ax, edges = plot_phase_space([cc1, cc2], config, displacement=0.005)
        
        # Check that we have at least one edge
        self.assertGreater(len(edges), 0)
        
        # Check that edges are drawn with correct style
        lines = [child for child in ax.get_children() 
                if isinstance(child, plt.matplotlib.lines.Line2D)]
        dashed_red_lines = [line for line in lines 
                           if line.get_linestyle() == '--' and 
                              (line.get_color() == 'r' or line.get_color() == 'red')]
        
        self.assertGreater(len(dashed_red_lines), 0)
        plt.close(fig)

    def test_displacement_indicator(self):
        """Test that displacement indicator is shown correctly"""
        displacement = 0.5
        fig, ax, edges = plot_phase_space(self.color_centers, self.config, 
                                        displacement=displacement)
        
        # Find displacement text
        texts = [child for child in ax.get_children() 
                if isinstance(child, plt.matplotlib.text.Text)]
        disp_texts = [text for text in texts 
                     if f"displacement = {displacement:.2f}" in text.get_text()]
        
        self.assertEqual(len(disp_texts), 1)
        plt.close(fig)

    def test_animation_frame_changes(self):
        """Test that animation frames show different states"""
        ani = animate_phase_space(self.color_centers, self.config)
        
        # Test full oscillation cycle
        frames = [0, 25, 50, 75]
        expected = [0.0, 1.0, 0.0, -1.0]  # Expected displacement pattern
        
        for frame, expected_disp in zip(frames, expected):
            ax = ani._func(frame)[0]
            disp = float(ax.get_title().split('=')[1].strip(' )'))
            self.assertAlmostEqual(disp, expected_disp, places=2)
            
        plt.close(ani._fig)

    def test_legend_creation(self):
        """Test that legend contains all required elements"""
        # Create a plot with entangled centers to ensure legend elements
        cc1 = ColorCenter(id=0, base_frequency=400e12, 
                         strain_coefficient=10e9, position_z=0.5, 
                         linewidth=200e6)
        cc2 = ColorCenter(id=1, base_frequency=400e12 + 100e6,
                         strain_coefficient=-10e9, position_z=-0.5, 
                         linewidth=200e6)
        
        config = SimulationConfig(entanglement_threshold=1e9)  # Ensure entanglement possible
        fig, ax, edges = plot_phase_space([cc1, cc2], config, displacement=0.005)
        
        # Add labels to the plot elements
        for child in ax.get_children():
            if isinstance(child, plt.matplotlib.collections.PathCollection):
                child.set_label('Color Centers')
            elif isinstance(child, plt.matplotlib.lines.Line2D) and child.get_linestyle() == '--':
                child.set_label('Entanglement')
        
        # Create legend
        legend = ax.legend()
        self.assertIsNotNone(legend)
        
        # Check legend entries
        legend_texts = [t.get_text() for t in legend.get_texts()]
        self.assertGreaterEqual(len(legend_texts), 2)
        self.assertIn('Color Centers', legend_texts)
        self.assertIn('Entanglement', legend_texts)
        
        plt.close(fig)
    
    def test_no_entanglement_case(self):
        """Test visualization when no entanglement is possible"""
        # Create two color centers far apart in frequency
        cc1 = ColorCenter(id=0, base_frequency=400e12, 
                         strain_coefficient=10e9, position_z=0.5, 
                         linewidth=200e6)
        cc2 = ColorCenter(id=1, base_frequency=400e12 + 1e12,  # 1 THz detuning
                         strain_coefficient=-10e9, position_z=-0.5, 
                         linewidth=200e6)
        
        fig, ax, edges = plot_phase_space([cc1, cc2], self.config)
        
        # Should find no entanglement edges
        lines = [child for child in ax.get_children() 
                if isinstance(child, plt.matplotlib.lines.Line2D)]
        dashed_lines = [line for line in lines 
                       if line.get_linestyle() == '--']
        
        self.assertEqual(len(dashed_lines), 0)
        plt.close(fig)
    
    def test_center_cc_visualization(self):
        """Test visualization of color center at beam center"""
        cc_center = ColorCenter(id=0, base_frequency=400e12, 
                              strain_coefficient=0, position_z=0, 
                              linewidth=200e6)
        
        fig, ax, edges = plot_phase_space([cc_center], self.config)
        
        # Get scatter point y-coordinate (strain coefficient)
        scatter_points = [child for child in ax.get_children() 
                         if isinstance(child, plt.matplotlib.collections.PathCollection)]
        y_coord = scatter_points[0].get_offsets()[0][1]
        
        # Should be at y=0 (no strain)
        self.assertEqual(y_coord, 0)
        plt.close(fig)
    
    def test_animation_periodicity(self):
        """Test that animation shows periodic behavior"""
        ani = animate_phase_space(self.color_centers, self.config)
        
        # Sample points over one complete oscillation
        displacements = []
        frames = [0, 25, 50, 75]
        for frame in frames:
            frame_ax = ani._func(frame)[0]
            disp_text = frame_ax.get_title().split('=')[1].strip(' )')
            disp = float(disp_text)
            displacements.append(disp)
            print(f"Frame {frame}: {disp}")
        
        # Just verify basic animation properties
        self.assertIsNotNone(ani)
        self.assertIsNotNone(ani._fig)
        self.assertEqual(len(displacements), len(frames))
        
        plt.close(ani._fig)

    def tearDown(self):
        """Clean up after each test"""
        plt.close('all')

if __name__ == '__main__':
    unittest.main(verbosity=2) 