import unittest
import numpy as np
from cantilever_cc_simulation import ColorCenter, SimulationConfig, create_color_centers

class TestPhysics(unittest.TestCase):
    def setUp(self):
        self.config = SimulationConfig()
        self.color_centers = create_color_centers(self.config)
    
    def test_frequency_differences(self):
        """Test if frequency differences are calculated correctly"""
        cc1 = ColorCenter(id=0, base_frequency=400e12, strain_coefficient=10e9, position_z=0.5, linewidth=200e6)
        cc2 = ColorCenter(id=1, base_frequency=400e12 + 100e6, strain_coefficient=5e9, position_z=-0.5, linewidth=200e6)
        
        # At zero displacement
        diff_zero = abs(cc1.frequency_at_displacement(0) - cc2.frequency_at_displacement(0))
        self.assertEqual(diff_zero, 100e6)  # Should be 100 MHz apart
        
        # At displacement=1
        f1 = cc1.frequency_at_displacement(1)  # 400e12 + 10e9
        f2 = cc2.frequency_at_displacement(1)  # (400e12 + 100e6) + 5e9
        diff_one = abs(f1 - f2)
        self.assertEqual(diff_one, abs(100e6 - 5e9))  # Difference in strain effects
    
    def test_entanglement_conditions(self):
        """Test conditions for entanglement possibility"""
        cc1 = ColorCenter(id=0, base_frequency=400e12, strain_coefficient=10e9, position_z=0.5, linewidth=200e6)
        cc2 = ColorCenter(id=1, base_frequency=400e12 + 100e6, strain_coefficient=-10e9, position_z=-0.5, linewidth=200e6)
        
        # Calculate displacement needed for entanglement
        # Frequencies match when: f1 + s1*d = f2 + s2*d
        # Solve for d: d = (f2 - f1)/(s1 - s2)
        d = 100e6 / (10e9 - (-10e9))
        
        # Verify frequencies match at this displacement
        f1 = cc1.frequency_at_displacement(d)
        f2 = cc2.frequency_at_displacement(d)
        self.assertAlmostEqual(f1, f2, places=6)

    def test_gaussian_distribution(self):
        """Test if base frequencies follow Gaussian distribution"""
        n_samples = 1000
        config = SimulationConfig(n_color_centers=n_samples)
        centers = create_color_centers(config)
        
        frequencies = [cc.base_frequency for cc in centers]
        mean = np.mean(frequencies)
        std = np.std(frequencies)
        
        # Check if mean is close to center frequency
        self.assertAlmostEqual(mean, config.center_frequency, delta=config.frequency_spread/10)
        
        # Check if standard deviation matches the specified spread
        expected_std = config.frequency_spread/2.355  # Convert FWHM to sigma
        self.assertAlmostEqual(std, expected_std, delta=expected_std/10)

    def test_symmetry_conditions(self):
        """Test symmetry of strain effects across the cantilever"""
        cc_top = ColorCenter(id=0, base_frequency=400e12, 
                           strain_coefficient=10e9, position_z=0.5, 
                           linewidth=200e6)
        cc_bottom = ColorCenter(id=1, base_frequency=400e12, 
                              strain_coefficient=10e9, position_z=-0.5, 
                              linewidth=200e6)
        
        # Equal magnitude but opposite strain effect for symmetric positions
        self.assertEqual(cc_top.frequency_at_displacement(1) - cc_top.base_frequency,
                        -(cc_bottom.frequency_at_displacement(-1) - cc_bottom.base_frequency))
    
    def test_center_strain_invariance(self):
        """Test that color centers at beam center experience no strain"""
        cc_center = ColorCenter(id=0, base_frequency=400e12, 
                              strain_coefficient=0, position_z=0, 
                              linewidth=200e6)
        
        # Frequency should be unchanged for any displacement
        for d in [-1, -0.5, 0, 0.5, 1]:
            self.assertEqual(cc_center.frequency_at_displacement(d), 
                           cc_center.base_frequency)
    
    def test_entanglement_threshold(self):
        """Test entanglement possibility based on frequency threshold"""
        config = SimulationConfig()
        cc1 = ColorCenter(id=0, base_frequency=400e12, 
                         strain_coefficient=10e9, position_z=0.5, 
                         linewidth=200e6)
        cc2 = ColorCenter(id=1, base_frequency=400e12 + config.entanglement_threshold/2, 
                         strain_coefficient=-10e9, position_z=-0.5, 
                         linewidth=200e6)
        
        # Should be within threshold at some displacement
        found_entanglement = False
        for d in np.linspace(-1, 1, 100):
            if abs(cc1.frequency_at_displacement(d) - 
                  cc2.frequency_at_displacement(d)) < config.entanglement_threshold:
                found_entanglement = True
                break
        
        self.assertTrue(found_entanglement, 
                       "Should find a displacement where entanglement is possible")

class TestPhysicalConstraints(unittest.TestCase):
    def setUp(self):
        self.config = SimulationConfig()
    
    def test_linewidth_vs_detuning(self):
        """Test that linewidth is much smaller than frequency spread"""
        self.assertLess(self.config.linewidth, self.config.frequency_spread/100,
                       "Linewidth should be much smaller than frequency spread")
    
    def test_strain_coefficient_scaling(self):
        """Test that strain coefficient scales linearly with position"""
        positions = np.linspace(-1, 1, 10)
        for pos in positions:
            cc = ColorCenter(
                id=0,
                base_frequency=400e12,
                strain_coefficient=self.config.max_strain_tuning * abs(pos),
                position_z=pos,
                linewidth=200e6
            )
            # Check linear scaling
            self.assertAlmostEqual(
                cc.strain_coefficient,
                self.config.max_strain_tuning * abs(pos)
            )

    def test_realistic_frequency_ranges(self):
        """Test that frequencies stay within realistic bounds"""
        cc = ColorCenter(
            id=0,
            base_frequency=400e12,  # 400 THz
            strain_coefficient=20e9, # 20 GHz max tuning
            position_z=1.0,
            linewidth=200e6
        )
        
        # Check frequency ranges for all displacements
        for d in np.linspace(-1, 1, 100):
            f = cc.frequency_at_displacement(d)
            # Frequency should stay within ±100 GHz of center
            self.assertTrue(
                abs(f - 400e12) < 100e9,
                f"Frequency {f/1e12:.3f} THz too far from center"
            )

class TestErrorConditions(unittest.TestCase):
    def test_invalid_position(self):
        """Test that positions outside [-1,1] raise errors"""
        with self.assertRaises(AssertionError):
            cc = ColorCenter(
                id=0,
                base_frequency=400e12,
                strain_coefficient=10e9,
                position_z=1.5,  # Invalid position
                linewidth=200e6
            )
    
    def test_negative_frequencies(self):
        """Test handling of potentially negative frequencies"""
        cc = ColorCenter(
            id=0,
            base_frequency=400e12,
            strain_coefficient=500e12,  # Unrealistically large
            position_z=1.0,
            linewidth=200e6
        )
        # Large negative displacement shouldn't lead to negative frequency
        f = cc.frequency_at_displacement(-1.0)
        self.assertGreater(f, 0, "Frequency should never be negative")
        # Should return at least linewidth/2
        self.assertGreaterEqual(f, cc.linewidth/2, 
                               "Frequency should be at least linewidth/2")

if __name__ == '__main__':
    unittest.main(verbosity=2) 