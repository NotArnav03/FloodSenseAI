"""
Post-processing utilities for flood mask refinement.

Applied after model inference to clean up predictions
and extract actionable flood zone information.
"""

import numpy as np
import cv2


def refine_mask(mask, min_area_ratio=0.001, morph_kernel_size=7):
    """Apply morphological operations and remove small components.

    Args:
        mask: float32 array [0, 1] of flood probabilities
        min_area_ratio: minimum component area as fraction of total image
        morph_kernel_size: kernel size for morphological operations

    Returns:
        Refined float32 mask [0, 1]
    """
    h, w = mask.shape
    min_area = int(h * w * min_area_ratio)

    # Binarize for morphological ops
    binary = (mask > 0.3).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (morph_kernel_size, morph_kernel_size))

    # Close gaps, then remove noise
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    # Remove small connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            binary[labels == i] = 0

    # Fill holes within flood regions
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(binary, contours, -1, 1, thickness=cv2.FILLED)

    # Smooth edges
    refined = cv2.GaussianBlur(binary.astype(np.float32), (5, 5), 0)

    # Blend with original soft probabilities for smooth boundaries
    result = 0.7 * refined + 0.3 * mask
    return np.clip(result, 0.0, 1.0).astype(np.float32)


def extract_flood_zones(mask, threshold=0.5):
    """Extract individual flood zones with area and severity stats.

    Args:
        mask: float32 array [0, 1] of flood probabilities
        threshold: binarization threshold

    Returns:
        List of dicts with zone_id, area_pixels, centroid, mean_severity, bbox
    """
    binary = (mask > threshold).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    zones = []
    for i in range(1, num_labels):  # skip background (0)
        zone_mask = (labels == i)
        severity = mask[zone_mask].mean()

        zones.append({
            "zone_id": i,
            "area_pixels": int(stats[i, cv2.CC_STAT_AREA]),
            "centroid": (float(centroids[i][0]), float(centroids[i][1])),
            "mean_severity": float(severity),
            "bbox": {
                "x": int(stats[i, cv2.CC_STAT_LEFT]),
                "y": int(stats[i, cv2.CC_STAT_TOP]),
                "w": int(stats[i, cv2.CC_STAT_WIDTH]),
                "h": int(stats[i, cv2.CC_STAT_HEIGHT]),
            },
        })

    # Sort by severity (worst first)
    zones.sort(key=lambda z: z["mean_severity"], reverse=True)
    return zones


def compute_flood_statistics(mask, threshold=0.5):
    """Compute overall flood statistics from a probability mask.

    Args:
        mask: float32 array [0, 1] of flood probabilities
        threshold: binarization threshold

    Returns:
        Dict with coverage_pct, mean_severity, max_severity, affected_area
    """
    binary = (mask > threshold).astype(np.float32)
    total_pixels = mask.size
    flood_pixels = binary.sum()

    flood_values = mask[mask > threshold]

    return {
        "coverage_pct": float(flood_pixels / total_pixels * 100),
        "affected_pixels": int(flood_pixels),
        "total_pixels": int(total_pixels),
        "mean_severity": float(flood_values.mean()) if len(flood_values) > 0 else 0.0,
        "max_severity": float(flood_values.max()) if len(flood_values) > 0 else 0.0,
        "num_zones": len(extract_flood_zones(mask, threshold)),
    }
