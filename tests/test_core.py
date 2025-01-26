import unittest
import numpy as np
from cantilever_cc_simulation import ColorCenter, SimulationConfig, create_color_centers, animate_phase_space
import matplotlib.pyplot as plt

class TestColorCenter(unittest.TestCase):
    def setUp(self):
        self.cc = ColorCenter(
            id=0,
            base_frequency=400e12,  # 400 THz
            strain_coefficient=10e9,  # 10 GHz
            position_z=0.5,
            linewidth=200e6  # 200 MHz
        )
    
    def test_frequency_at_displacement(self):
        """Test frequency calculation at different displacements"""
        # At zero displacement, should return base frequency
        self.assertEqual(self.cc.frequency_at_displacement(0), 400e12)
        
        # At displacement=1, should add strain_coefficient
        self.assertEqual(self.cc.frequency_at_displacement(1), 400e12 + 10e9)
        
        # At displacement=-1, should subtract strain_coefficient
        self.assertEqual(self.cc.frequency_at_displacement(-1), 400e12 - 10e9)

class TestSimulationConfig(unittest.TestCase):
    def setUp(self):
        self.config = SimulationConfig()
    
    def test_default_values(self):
        """Test default configuration values"""
        self.assertEqual(self.config.n_color_centers, 10)
        self.assertEqual(self.config.center_frequency, 400e12)
        self.assertEqual(self.config.linewidth, 200e6)
        self.assertEqual(self.config.random_seed, 42)
    
    def test_validation(self):
        """Test configuration validation"""
        # Test invalid probability
        with self.assertRaises(AssertionError):
            invalid_config = SimulationConfig(max_displacement=2.0)
            invalid_config.validate()
        
        # Test invalid number of color centers
        with self.assertRaises(AssertionError):
            invalid_config = SimulationConfig(n_color_centers=0)
            invalid_config.validate()

class TestColorCenterCreation(unittest.TestCase):
    def setUp(self):
        self.config = SimulationConfig(
            n_color_centers=5,
            random_seed=42
        )
        self.color_centers = create_color_centers(self.config)
    
    def test_number_of_centers(self):
        """Test if correct number of color centers are created"""
        self.assertEqual(len(self.color_centers), self.config.n_color_centers)
    
    def test_position_range(self):
        """Test if positions are within valid range"""
        for cc in self.color_centers:
            self.assertTrue(-1 <= cc.position_z <= 1)
    
    def test_strain_coefficient_scaling(self):
        """Test if strain coefficients scale with position"""
        for cc in self.color_centers:
            expected_max = self.config.max_strain_tuning * abs(cc.position_z)
            self.assertEqual(cc.strain_coefficient, expected_max)
    
    def test_reproducibility(self):
        """Test if random seed produces reproducible results"""
        color_centers2 = create_color_centers(self.config)
        for cc1, cc2 in zip(self.color_centers, color_centers2):
            self.assertEqual(cc1.base_frequency, cc2.base_frequency)
            self.assertEqual(cc1.position_z, cc2.position_z)

if __name__ == '__main__':
    unittest.main(verbosity=2) 