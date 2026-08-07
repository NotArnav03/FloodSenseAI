import numpy as np

class AllocationEngine:
    def __init__(self, risk_map, resources):
        self.risk_map = risk_map
        self.resources = resources

    def allocate(self, top_k=20):
        """
        Allocate resources to top-k highest risk zones.
        Resources are distributed proportionally based on relative risk scores.
        """
        flat_risk = self.risk_map.flatten()
        
        # Get top-k zones
        sorted_indices = np.argsort(flat_risk)[::-1][:top_k]
        top_risks = flat_risk[sorted_indices]
        
        # Normalize to get proportions among top zones only
        total_top_risk = np.sum(top_risks)
        if total_top_risk == 0:
            proportions = np.ones(top_k) / top_k
        else:
            proportions = top_risks / total_top_risk

        allocation_plan = []

        for idx, proportion in zip(sorted_indices, proportions):
            coord = np.unravel_index(idx, self.risk_map.shape)
            zone_allocation = {
                "zone_index": int(idx),
                "coordinates": (int(coord[0]), int(coord[1])),
                "risk_score": float(flat_risk[idx]),
                "proportion": float(proportion),
                "food_packets": int(proportion * self.resources["food_packets"]),
                "medical_kits": int(proportion * self.resources["medical_kits"]),
                "boats": int(proportion * self.resources.get("boats", 0))
            }
            allocation_plan.append(zone_allocation)

        return allocation_plan
    
    def get_cluster_allocation(self, cluster_size=10):
        """
        Allocate resources to clusters of high-risk zones.
        Groups nearby high-risk cells into clusters for more practical allocation.
        """
        from scipy import ndimage
        
        # Threshold to identify high-risk areas
        threshold = np.percentile(self.risk_map, 95)
        high_risk_mask = self.risk_map > threshold
        
        # Label connected components
        labeled, num_clusters = ndimage.label(high_risk_mask)
        
        cluster_allocations = []
        for i in range(1, num_clusters + 1):
            cluster_mask = labeled == i
            cluster_risk = np.sum(self.risk_map[cluster_mask])
            cluster_size_count = np.sum(cluster_mask)
            centroid = ndimage.center_of_mass(cluster_mask)
            
            cluster_allocations.append({
                "cluster_id": i,
                "centroid": (int(centroid[0]), int(centroid[1])),
                "size": int(cluster_size_count),
                "total_risk": float(cluster_risk),
                "mean_risk": float(cluster_risk / cluster_size_count)
            })
        
        # Sort by total risk and allocate
        cluster_allocations.sort(key=lambda x: x['total_risk'], reverse=True)
        total_cluster_risk = sum(c['total_risk'] for c in cluster_allocations)
        
        for cluster in cluster_allocations:
            if total_cluster_risk > 0:
                proportion = cluster['total_risk'] / total_cluster_risk
            else:
                proportion = 0
            cluster['food_packets'] = int(proportion * self.resources["food_packets"])
            cluster['medical_kits'] = int(proportion * self.resources["medical_kits"])
            cluster['boats'] = int(proportion * self.resources.get("boats", 0))
        
        return cluster_allocations