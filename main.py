import numpy as np
from core.risk_engine import RiskEngine
from core.allocation_engine import AllocationEngine


def main():
    print("\n--- Disaster Intelligence System ---\n")

    # Initialize Risk Engine
    engine = RiskEngine()
    engine.load_flood_mask()
    engine.generate_population_grid()
    engine.compute_risk()

    # Convert top zones to coordinates
    top_indices = engine.get_top_risk_zones(top_k=10)
    coords = np.unravel_index(top_indices, engine.risk_map.shape)

    print("\nTop 10 High-Risk Zones (row, col):")
    for r, c in zip(coords[0], coords[1]):
        print(f"Zone: ({r}, {c}) | Risk Score: {engine.risk_map[r][c]:.4f}")

    # Define available resources
    resources = {
        "food_packets": 5000,
        "medical_kits": 1200,
        "boats": 50
    }

    # Initialize Allocation Engine
    allocator = AllocationEngine(engine.risk_map, resources)
    allocation_plan = allocator.allocate()

    print("\n--- Allocation Plan (Top Zones) ---")
    for zone in allocation_plan[:10]:
        print(zone)

    print("\nSystem execution completed.\n")


if __name__ == "__main__":
    main()