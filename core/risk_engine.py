import numpy as np

class RiskEngine:
    def __init__(self, flood_mask_path=None):
        self.flood_mask_path = flood_mask_path
        self.flood_mask = None
        self.population_grid = None
        self.risk_map = None

    def load_flood_mask(self):
        if self.flood_mask_path:
            self.flood_mask = np.load(self.flood_mask_path)
        else:
            # Temporary synthetic flood map for testing
            self.flood_mask = np.random.rand(100, 100)

        print("Flood mask loaded. Shape:", self.flood_mask.shape)

    def generate_population_grid(self):
        # Simulated population density
        self.population_grid = np.random.rand(
            self.flood_mask.shape[0],
            self.flood_mask.shape[1]
        )

        print("Population grid generated.")

    def compute_risk(self):
        # Normalize both grids
        flood_norm = self.flood_mask / np.max(self.flood_mask)
        pop_norm = self.population_grid / np.max(self.population_grid)

        self.risk_map = flood_norm * pop_norm

        print("Risk map computed.")

    def get_top_risk_zones(self, top_k=10):
        flat = self.risk_map.flatten()
        indices = np.argsort(flat)[-top_k:]
        return indices