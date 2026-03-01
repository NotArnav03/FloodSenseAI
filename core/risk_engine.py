import numpy as np
import os
from pathlib import Path

class RiskEngine:
    def __init__(self, data_dir=None, flood_mask_path=None):
        self.data_dir = data_dir or Path(__file__).parent.parent / "data"
        self.flood_mask_path = flood_mask_path
        self.flood_mask = None
        self.population_grid = None
        self.risk_map = None
        self.current_map_name = None

    def list_available_maps(self):
        """List all available flood probability maps in the data directory."""
        if not os.path.exists(self.data_dir):
            return []
        return [f for f in os.listdir(self.data_dir) if f.endswith('.npy')]

    def get_maps_by_region(self):
        """Group available maps by region/country."""
        maps = self.list_available_maps()
        regions = {}
        for m in maps:
            region = m.split('_')[0]
            if region not in regions:
                regions[region] = []
            regions[region].append(m)
        return regions

    def load_flood_mask(self, map_name=None):
        """
        Load a flood probability map.
        Args:
            map_name: Specific map filename to load. If None, loads from flood_mask_path
                     or uses the first available map.
        """
        if map_name:
            self.flood_mask_path = os.path.join(self.data_dir, map_name)
            self.current_map_name = map_name
        
        if self.flood_mask_path and os.path.exists(self.flood_mask_path):
            self.flood_mask = np.load(self.flood_mask_path)
            self.current_map_name = os.path.basename(self.flood_mask_path)
        else:
            # Try to load first available map
            available = self.list_available_maps()
            if available:
                self.flood_mask_path = os.path.join(self.data_dir, available[0])
                self.flood_mask = np.load(self.flood_mask_path)
                self.current_map_name = available[0]
            else:
                # Fallback to synthetic flood map for testing
                print("No flood maps found. Using synthetic data.")
                self.flood_mask = np.random.rand(512, 512).astype(np.float32)
                self.current_map_name = "synthetic"

        print(f"Flood mask loaded: {self.current_map_name}")
        print(f"  Shape: {self.flood_mask.shape}")
        print(f"  Min: {self.flood_mask.min():.4f}, Max: {self.flood_mask.max():.4f}")

    def generate_population_grid(self, method='uniform'):
        """
        Generate population density grid.
        Args:
            method: 'uniform' for random, 'gradient' for urban-to-rural simulation
        """
        h, w = self.flood_mask.shape
        
        if method == 'gradient':
            # Simulate higher population in center (urban core)
            x = np.linspace(-1, 1, w)
            y = np.linspace(-1, 1, h)
            xx, yy = np.meshgrid(x, y)
            self.population_grid = np.exp(-(xx**2 + yy**2) / 0.5)
            self.population_grid += np.random.rand(h, w) * 0.2  # Add noise
        else:
            self.population_grid = np.random.rand(h, w).astype(np.float32)

        print("Population grid generated.")

    def compute_risk(self, weights=None):
        """
        Compute risk map from flood probability and population density.
        Args:
            weights: Dict with 'flood' and 'population' weights. Default is equal weighting.
        """
        if weights is None:
            weights = {'flood': 0.6, 'population': 0.4}
        
        # Normalize both grids to 0-1
        flood_norm = self.flood_mask / (np.max(self.flood_mask) + 1e-8)
        pop_norm = self.population_grid / (np.max(self.population_grid) + 1e-8)

        # Weighted combination
        self.risk_map = (weights['flood'] * flood_norm + 
                         weights['population'] * pop_norm * flood_norm)

        print(f"Risk map computed. Max risk: {self.risk_map.max():.4f}")

    def get_top_risk_zones(self, top_k=10):
        """Get indices of top-k highest risk zones."""
        flat = self.risk_map.flatten()
        indices = np.argsort(flat)[-top_k:][::-1]  # Sorted descending
        return indices

    def get_risk_statistics(self):
        """Get comprehensive statistics about the risk map."""
        if self.risk_map is None:
            return None
        
        return {
            'map_name': self.current_map_name,
            'shape': self.risk_map.shape,
            'total_cells': self.risk_map.size,
            'min_risk': float(self.risk_map.min()),
            'max_risk': float(self.risk_map.max()),
            'mean_risk': float(self.risk_map.mean()),
            'std_risk': float(self.risk_map.std()),
            'high_risk_cells': int(np.sum(self.risk_map > 0.5)),
            'critical_cells': int(np.sum(self.risk_map > 0.8))
        }