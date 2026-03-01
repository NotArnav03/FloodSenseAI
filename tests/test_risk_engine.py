"""
Unit tests for the Edge-Drive Flood Risk Intelligence System.
"""
import pytest
import numpy as np
import os
import sys
import tempfile

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.risk_engine import RiskEngine
from core.allocation_engine import AllocationEngine


class TestRiskEngine:
    """Tests for RiskEngine class."""
    
    def test_initialization(self):
        """Test RiskEngine initializes correctly."""
        engine = RiskEngine()
        assert engine.flood_mask is None
        assert engine.population_grid is None
        assert engine.risk_map is None
    
    def test_initialization_with_custom_path(self):
        """Test RiskEngine with custom data directory."""
        engine = RiskEngine(data_dir="/custom/path")
        assert engine.data_dir == "/custom/path"
    
    def test_load_synthetic_flood_mask(self):
        """Test loading synthetic flood mask when no files available."""
        engine = RiskEngine(data_dir="/nonexistent/path")
        engine.load_flood_mask()
        
        assert engine.flood_mask is not None
        assert engine.flood_mask.shape == (512, 512)
        assert engine.current_map_name == "synthetic"
    
    def test_load_flood_mask_from_file(self):
        """Test loading flood mask from actual .npy file."""
        # Create temporary npy file
        with tempfile.TemporaryDirectory() as tmpdir:
            test_data = np.random.rand(256, 256).astype(np.float32)
            test_file = os.path.join(tmpdir, "Test_123_flood_prob.npy")
            np.save(test_file, test_data)
            
            engine = RiskEngine(data_dir=tmpdir)
            engine.load_flood_mask("Test_123_flood_prob.npy")
            
            assert engine.flood_mask is not None
            assert engine.flood_mask.shape == (256, 256)
            assert engine.current_map_name == "Test_123_flood_prob.npy"
            np.testing.assert_array_equal(engine.flood_mask, test_data)
    
    def test_list_available_maps(self):
        """Test listing available maps."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            np.save(os.path.join(tmpdir, "India_001_flood_prob.npy"), np.zeros((10, 10)))
            np.save(os.path.join(tmpdir, "USA_002_flood_prob.npy"), np.zeros((10, 10)))
            
            engine = RiskEngine(data_dir=tmpdir)
            maps = engine.list_available_maps()
            
            assert len(maps) == 2
            assert "India_001_flood_prob.npy" in maps
            assert "USA_002_flood_prob.npy" in maps
    
    def test_get_maps_by_region(self):
        """Test grouping maps by region."""
        with tempfile.TemporaryDirectory() as tmpdir:
            np.save(os.path.join(tmpdir, "India_001_flood_prob.npy"), np.zeros((10, 10)))
            np.save(os.path.join(tmpdir, "India_002_flood_prob.npy"), np.zeros((10, 10)))
            np.save(os.path.join(tmpdir, "USA_001_flood_prob.npy"), np.zeros((10, 10)))
            
            engine = RiskEngine(data_dir=tmpdir)
            regions = engine.get_maps_by_region()
            
            assert "India" in regions
            assert "USA" in regions
            assert len(regions["India"]) == 2
            assert len(regions["USA"]) == 1
    
    def test_generate_population_grid_uniform(self):
        """Test uniform population grid generation."""
        engine = RiskEngine()
        engine.flood_mask = np.ones((100, 100))
        engine.generate_population_grid(method='uniform')
        
        assert engine.population_grid is not None
        assert engine.population_grid.shape == (100, 100)
        assert engine.population_grid.min() >= 0
        assert engine.population_grid.max() <= 1
    
    def test_generate_population_grid_gradient(self):
        """Test gradient population grid generation."""
        engine = RiskEngine()
        engine.flood_mask = np.ones((100, 100))
        engine.generate_population_grid(method='gradient')
        
        assert engine.population_grid is not None
        assert engine.population_grid.shape == (100, 100)
        # Center should have higher values than corners (urban core)
        center_val = engine.population_grid[50, 50]
        corner_val = engine.population_grid[0, 0]
        assert center_val > corner_val
    
    def test_compute_risk(self):
        """Test risk map computation."""
        engine = RiskEngine()
        engine.flood_mask = np.ones((100, 100)) * 0.5
        engine.population_grid = np.ones((100, 100)) * 0.5
        engine.compute_risk()
        
        assert engine.risk_map is not None
        assert engine.risk_map.shape == (100, 100)
        assert engine.risk_map.min() >= 0
    
    def test_compute_risk_with_custom_weights(self):
        """Test risk computation with custom weights."""
        engine = RiskEngine()
        engine.flood_mask = np.ones((50, 50))
        engine.population_grid = np.ones((50, 50))
        
        engine.compute_risk(weights={'flood': 0.8, 'population': 0.2})
        
        assert engine.risk_map is not None
    
    def test_get_top_risk_zones(self):
        """Test extracting top risk zones."""
        engine = RiskEngine()
        engine.risk_map = np.zeros((10, 10))
        engine.risk_map[5, 5] = 1.0
        engine.risk_map[3, 3] = 0.9
        engine.risk_map[7, 7] = 0.8
        
        top_zones = engine.get_top_risk_zones(top_k=3)
        
        assert len(top_zones) == 3
        # First should be highest risk (5, 5)
        assert top_zones[0] == 55  # 5*10 + 5
    
    def test_get_risk_statistics(self):
        """Test risk statistics generation."""
        engine = RiskEngine()
        engine.current_map_name = "test_map.npy"
        engine.risk_map = np.random.rand(100, 100)
        
        stats = engine.get_risk_statistics()
        
        assert stats is not None
        assert 'map_name' in stats
        assert 'shape' in stats
        assert 'mean_risk' in stats
        assert 'max_risk' in stats
        assert 'high_risk_cells' in stats
        assert 'critical_cells' in stats


class TestAllocationEngine:
    """Tests for AllocationEngine class."""
    
    def test_initialization(self):
        """Test AllocationEngine initialization."""
        risk_map = np.random.rand(10, 10)
        resources = {"food_packets": 100, "medical_kits": 50, "boats": 10}
        
        allocator = AllocationEngine(risk_map, resources)
        
        assert allocator.risk_map is not None
        assert allocator.resources == resources
    
    def test_allocate_basic(self):
        """Test basic resource allocation."""
        risk_map = np.random.rand(50, 50)
        resources = {"food_packets": 1000, "medical_kits": 500, "boats": 20}
        
        allocator = AllocationEngine(risk_map, resources)
        plan = allocator.allocate(top_k=5)
        
        assert len(plan) == 5
        assert all('zone_index' in zone for zone in plan)
        assert all('coordinates' in zone for zone in plan)
        assert all('food_packets' in zone for zone in plan)
        assert all('medical_kits' in zone for zone in plan)
        assert all('boats' in zone for zone in plan)
    
    def test_allocate_proportional_distribution(self):
        """Test that resources are distributed proportionally."""
        risk_map = np.zeros((10, 10))
        risk_map[0, 0] = 1.0  # Highest risk
        risk_map[1, 1] = 0.5  # Half the risk
        
        resources = {"food_packets": 1000, "medical_kits": 500, "boats": 10}
        
        allocator = AllocationEngine(risk_map, resources)
        plan = allocator.allocate(top_k=2)
        
        # Higher risk zone should get more resources
        assert plan[0]['food_packets'] > plan[1]['food_packets']
        assert plan[0]['risk_score'] > plan[1]['risk_score']
    
    def test_allocate_total_resources(self):
        """Test that total allocated resources don't exceed available."""
        risk_map = np.random.rand(20, 20)
        resources = {"food_packets": 100, "medical_kits": 50, "boats": 5}
        
        allocator = AllocationEngine(risk_map, resources)
        plan = allocator.allocate(top_k=10)
        
        total_food = sum(zone['food_packets'] for zone in plan)
        total_medical = sum(zone['medical_kits'] for zone in plan)
        total_boats = sum(zone['boats'] for zone in plan)
        
        # Due to rounding, should be close to but not exceed resources
        assert total_food <= resources['food_packets']
        assert total_medical <= resources['medical_kits']
        assert total_boats <= resources['boats']
    
    def test_allocate_with_zero_risk(self):
        """Test allocation with zero risk map."""
        risk_map = np.zeros((10, 10))
        resources = {"food_packets": 100, "medical_kits": 50, "boats": 5}
        
        allocator = AllocationEngine(risk_map, resources)
        plan = allocator.allocate(top_k=5)
        
        # Should still return zones, with equal distribution
        assert len(plan) == 5


class TestIntegration:
    """Integration tests for the complete workflow."""
    
    def test_full_pipeline_synthetic(self):
        """Test complete pipeline with synthetic data."""
        engine = RiskEngine()
        engine.load_flood_mask()
        engine.generate_population_grid()
        engine.compute_risk()
        
        assert engine.risk_map is not None
        
        resources = {"food_packets": 5000, "medical_kits": 1200, "boats": 50}
        allocator = AllocationEngine(engine.risk_map, resources)
        plan = allocator.allocate()
        
        assert len(plan) > 0
        assert all(zone['food_packets'] >= 0 for zone in plan)
    
    def test_full_pipeline_with_file(self):
        """Test complete pipeline with actual file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test flood map
            flood_data = np.random.rand(128, 128).astype(np.float32)
            flood_data[60:70, 60:70] = 0.95  # High risk area
            np.save(os.path.join(tmpdir, "Test_001_flood_prob.npy"), flood_data)
            
            engine = RiskEngine(data_dir=tmpdir)
            engine.load_flood_mask("Test_001_flood_prob.npy")
            engine.generate_population_grid(method='gradient')
            engine.compute_risk()
            
            stats = engine.get_risk_statistics()
            assert stats['high_risk_cells'] > 0
            
            resources = {"food_packets": 1000, "medical_kits": 200, "boats": 10}
            allocator = AllocationEngine(engine.risk_map, resources)
            plan = allocator.allocate(top_k=5)
            
            # Top risk zones should be in the high-risk area
            assert len(plan) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
