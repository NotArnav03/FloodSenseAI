import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

from core.risk_engine import RiskEngine
from core.allocation_engine import AllocationEngine


def visualize_risk_map(risk_map, title="Flood Risk Heatmap", save_path=None):
    """
    Displays a heatmap of the computed risk surface.
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(risk_map, cmap="inferno")
    plt.colorbar(label="Risk Intensity")
    plt.title(title)
    plt.xlabel("Longitude Grid")
    plt.ylabel("Latitude Grid")
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")
    else:
        plt.show()


def visualize_comparison(flood_mask, risk_map, map_name):
    """
    Side-by-side comparison of flood probability and risk map.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    im1 = axes[0].imshow(flood_mask, cmap="Blues")
    axes[0].set_title(f"Flood Probability\n{map_name}")
    plt.colorbar(im1, ax=axes[0], label="Probability")
    
    im2 = axes[1].imshow(risk_map, cmap="inferno")
    axes[1].set_title("Computed Risk Map")
    plt.colorbar(im2, ax=axes[1], label="Risk Intensity")
    
    plt.tight_layout()
    plt.show()


def list_available_regions(engine):
    """Print available regions and their map counts."""
    regions = engine.get_maps_by_region()
    print("\n========== Available Flood Probability Maps ==========\n")
    print(f"Total maps: {sum(len(v) for v in regions.values())}")
    print(f"Regions: {len(regions)}\n")
    
    for region, maps in sorted(regions.items()):
        print(f"  {region}: {len(maps)} maps")
    
    return regions


def process_single_map(engine, map_name, resources, visualize=True):
    """Process a single flood probability map."""
    print(f"\n---------- Processing: {map_name} ----------\n")
    
    # Load and process
    engine.load_flood_mask(map_name)
    engine.generate_population_grid(method='gradient')
    engine.compute_risk()
    
    # Get statistics
    stats = engine.get_risk_statistics()
    print(f"\nRisk Statistics:")
    print(f"  High risk cells (>0.5): {stats['high_risk_cells']:,}")
    print(f"  Critical cells (>0.8): {stats['critical_cells']:,}")
    print(f"  Mean risk: {stats['mean_risk']:.4f}")
    
    # Extract top risk zones
    top_indices = engine.get_top_risk_zones(top_k=10)
    coords = np.unravel_index(top_indices, engine.risk_map.shape)
    
    print("\nTop 10 High-Risk Zones:")
    print("-" * 40)
    for i, (r, c) in enumerate(zip(coords[0], coords[1])):
        risk_value = engine.risk_map[r][c]
        print(f"  {i+1}. Zone ({r:3d}, {c:3d}) | Risk: {risk_value:.4f}")
    
    # Allocation
    allocator = AllocationEngine(engine.risk_map, resources)
    allocation_plan = allocator.allocate()
    
    print("\nResource Allocation (Top 5 Zones):")
    print("-" * 60)
    print(f"{'Zone':<15} {'Risk':>8} {'Food':>8} {'Medical':>8} {'Boats':>6}")
    print("-" * 60)
    for zone in allocation_plan[:5]:
        coord = zone['coordinates']
        print(f"({coord[0]:3d}, {coord[1]:3d})      {zone['risk_score']:>8.4f} {zone['food_packets']:>8} {zone['medical_kits']:>8} {zone['boats']:>6}")
    
    # Visualization
    if visualize:
        visualize_comparison(engine.flood_mask, engine.risk_map, map_name)
    
    return stats, allocation_plan


def batch_process(engine, region=None, top_n=None, save_dir=None):
    """Process multiple maps and generate summary."""
    regions = engine.get_maps_by_region()
    
    if region and region in regions:
        maps_to_process = regions[region]
    else:
        maps_to_process = engine.list_available_maps()
    
    if top_n:
        maps_to_process = maps_to_process[:top_n]
    
    print(f"\n========== Batch Processing {len(maps_to_process)} Maps ==========\n")
    
    results = []
    for map_name in maps_to_process:
        engine.load_flood_mask(map_name)
        engine.generate_population_grid(method='gradient')
        engine.compute_risk()
        stats = engine.get_risk_statistics()
        results.append(stats)
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, map_name.replace('.npy', '_risk.png'))
            visualize_risk_map(engine.risk_map, title=map_name, save_path=save_path)
    
    # Summary
    print("\n========== Batch Processing Summary ==========\n")
    print(f"{'Map Name':<45} {'Mean Risk':>10} {'High Risk':>12} {'Critical':>10}")
    print("-" * 80)
    
    for r in sorted(results, key=lambda x: x['mean_risk'], reverse=True)[:20]:
        print(f"{r['map_name']:<45} {r['mean_risk']:>10.4f} {r['high_risk_cells']:>12,} {r['critical_cells']:>10,}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Disaster Intelligence System - Flood Risk Analysis")
    parser.add_argument('--list', action='store_true', help='List available flood maps')
    parser.add_argument('--map', type=str, help='Process specific map (filename)')
    parser.add_argument('--region', type=str, help='Process all maps from a specific region')
    parser.add_argument('--batch', action='store_true', help='Batch process all maps')
    parser.add_argument('--top', type=int, default=5, help='Number of maps to process in batch mode')
    parser.add_argument('--no-viz', action='store_true', help='Disable visualization')
    parser.add_argument('--save-dir', type=str, help='Directory to save visualizations')
    
    args = parser.parse_args()
    
    print("\n========== Disaster Intelligence System ==========\n")
    
    # Initialize engine
    engine = RiskEngine()
    
    # Define resources
    resources = {
        "food_packets": 5000,
        "medical_kits": 1200,
        "boats": 50
    }
    
    if args.list:
        list_available_regions(engine)
        return
    
    if args.batch or args.region:
        batch_process(engine, region=args.region, top_n=args.top, save_dir=args.save_dir)
        return
    
    if args.map:
        process_single_map(engine, args.map, resources, visualize=not args.no_viz)
        return
    
    # Default: Interactive mode - process first available map
    available = engine.list_available_maps()
    if available:
        regions = list_available_regions(engine)
        print(f"\nProcessing first available map: {available[0]}\n")
        process_single_map(engine, available[0], resources, visualize=not args.no_viz)
    else:
        print("No flood probability maps found in data directory.")
        print("Using synthetic data for demonstration...")
        engine.load_flood_mask()
        engine.generate_population_grid()
        engine.compute_risk()
        visualize_risk_map(engine.risk_map)
    
    print("\nSystem execution completed.\n")


if __name__ == "__main__":
    main()