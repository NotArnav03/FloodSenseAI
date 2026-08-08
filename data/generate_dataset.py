"""
EDGE-DRIVE Training Dataset Generator

Downloads real satellite imagery from ESRI World Imagery (public, no API key)
and uses OpenStreetMap water body polygons as ground truth masks.

This approach gives ACCURATE labels (not color-guessed pseudo-labels):
  1. Query Overpass API for water/flood polygon geometries
  2. Download the satellite tile for that location
  3. Rasterize the OSM polygon onto the tile pixel space -> binary mask
  4. Augment and split train/val

Usage:
    python data/generate_dataset.py --output data/flood_dataset --samples 300
"""

import os
import sys
import math
import time
import json
import random
import argparse
import requests
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

TILE_SIZE = 256  # ESRI tiles are 256x256

# ---------------------------------------------------------------------------
# Flood-prone + water-rich regions to query
# (min_lat, min_lon, max_lat, max_lon, zoom, label)
# ---------------------------------------------------------------------------
QUERY_REGIONS = [
    # Bangladesh - major river systems
    (23.2, 89.8, 23.8, 90.5, 14, "bangladesh_brahmaputra"),
    (22.5, 89.2, 23.0, 90.0, 14, "bangladesh_ganges_delta"),
    (23.8, 90.5, 24.3, 91.2, 14, "bangladesh_meghna"),
    # Pakistan - Indus River
    (27.0, 67.5, 27.8, 68.5, 13, "pakistan_indus"),
    (25.5, 68.0, 26.3, 69.0, 13, "pakistan_sindh"),
    # Vietnam - Mekong Delta
    (10.0, 105.5, 10.8, 106.5, 14, "vietnam_mekong"),
    (9.5, 105.0, 10.2, 106.0, 14, "vietnam_mekong_south"),
    # Myanmar - Irrawaddy
    (16.5, 95.0, 17.2, 96.0, 13, "myanmar_irrawaddy"),
    # India - Ganges and Brahmaputra
    (25.0, 85.0, 26.0, 86.5, 13, "india_ganges"),
    (26.5, 92.0, 27.2, 93.0, 13, "india_brahmaputra"),
    (22.0, 88.0, 22.8, 89.5, 14, "india_sundarban"),
    # Nigeria - Niger Delta
    (5.0, 5.5, 6.2, 7.0, 13, "nigeria_niger_delta"),
    # Mozambique - Zambezi Delta
    (-18.5, 35.5, -17.8, 36.5, 13, "mozambique_zambezi"),
    # Amazon
    (-4.5, -63.5, -3.5, -62.0, 12, "brazil_amazon"),
    (-2.5, -60.0, -1.5, -58.5, 12, "brazil_amazon_north"),
    # Mississippi / Louisiana
    (29.2, -91.5, 30.0, -90.5, 13, "usa_mississippi"),
    (29.8, -90.2, 30.5, -89.5, 13, "usa_louisiana"),
    # Nile Delta
    (30.5, 30.5, 31.5, 32.0, 13, "egypt_nile_delta"),
    # China - Yangtze + Dongting Lake
    (28.5, 112.5, 30.0, 114.0, 12, "china_dongting_lake"),
    (29.5, 116.0, 30.5, 117.5, 13, "china_poyang_lake"),
    # Cambodia - Tonle Sap
    (12.0, 103.5, 13.5, 105.0, 12, "cambodia_tonle_sap"),
    # Europe rivers for variety
    (46.0, 8.5, 47.5, 10.5, 13, "switzerland_lakes"),
    (47.5, 16.0, 48.5, 17.5, 13, "austria_danube"),
    # Great Lakes USA
    (41.5, -82.5, 42.5, -81.0, 12, "usa_lake_erie"),
    (43.5, -79.5, 44.5, -76.5, 12, "usa_lake_ontario"),
]


def lat_lon_to_tile(lat, lon, zoom):
    n = 2 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return x, y


def tile_to_lat_lon(x, y, zoom):
    """Return the (lat, lon) of the NW corner of a tile."""
    n = 2 ** zoom
    lon = x / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    lat = math.degrees(lat_rad)
    return lat, lon


def tile_bbox(x, y, zoom):
    """Return (min_lat, min_lon, max_lat, max_lon) for a tile."""
    nw_lat, nw_lon = tile_to_lat_lon(x, y, zoom)
    se_lat, se_lon = tile_to_lat_lon(x + 1, y + 1, zoom)
    return (se_lat, nw_lon, nw_lat, se_lon)  # (min_lat, min_lon, max_lat, max_lon)


def lat_lon_to_pixel(lat, lon, tile_x, tile_y, zoom, tile_size=256):
    """Convert lat/lon to pixel coordinates within a specific tile.
    
    Always uses TILE_SIZE (256) for the projection math since tiles are 256px,
    then scales to tile_size for the output canvas.
    """
    n = 2 ** zoom
    # Pixel coordinates in world space at native tile size (256)
    world_px_x = (lon + 180.0) / 360.0 * n * TILE_SIZE
    lat_rad = math.radians(lat)
    world_px_y = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n * TILE_SIZE
    # Tile-local pixel coordinates at native resolution
    px = world_px_x - tile_x * TILE_SIZE
    py = world_px_y - tile_y * TILE_SIZE
    # Scale to output canvas size
    scale = tile_size / TILE_SIZE
    return px * scale, py * scale


def query_water_bodies(bbox, max_results=30):
    """Query Overpass API for water body polygons in a bounding box."""
    min_lat, min_lon, max_lat, max_lon = bbox
    query = f"""
[out:json][timeout:20];
(
  way["natural"="water"]({min_lat},{min_lon},{max_lat},{max_lon});
  way["waterway"~"river|stream|canal"]({min_lat},{min_lon},{max_lat},{max_lon});
  relation["natural"="water"]({min_lat},{min_lon},{max_lat},{max_lon});
);
out geom {max_results};
"""
    try:
        r = requests.post(
            "https://overpass-api.de/api/interpreter",
            data=query,
            timeout=30,
        )
        if r.status_code == 200:
            return r.json().get("elements", [])
    except Exception:
        pass
    return []


def rasterize_osm_polygons(elements, tile_x, tile_y, zoom, tile_size=256):
    """Rasterize OSM way/relation geometries onto a tile as a binary mask."""
    mask = np.zeros((tile_size, tile_size), dtype=np.uint8)

    for elem in elements:
        geom = elem.get("geometry", [])
        if not geom:
            continue

        # Convert lat/lon to pixel coords
        pts = []
        for node in geom:
            px, py = lat_lon_to_pixel(node["lat"], node["lon"], tile_x, tile_y, zoom, tile_size)
            pts.append([int(px), int(py)])

        if len(pts) < 3:
            continue

        pts_np = np.array(pts, dtype=np.int32)

        # Clip to tile bounds
        pts_np[:, 0] = np.clip(pts_np[:, 0], 0, tile_size - 1)
        pts_np[:, 1] = np.clip(pts_np[:, 1], 0, tile_size - 1)

        # Fill polygon
        cv2.fillPoly(mask, [pts_np], 255)
        # Also draw as thick polyline for rivers/canals
        if elem.get("tags", {}).get("waterway") in ("river", "canal"):
            cv2.polylines(mask, [pts_np], False, 255, thickness=max(2, tile_size // 64))

    return mask


def download_tile(x, y, z, session, retries=3):
    """Download a satellite tile from ESRI World Imagery."""
    url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
    for attempt in range(retries):
        try:
            r = session.get(url, timeout=20)
            if r.status_code == 200 and "image" in r.headers.get("content-type", ""):
                arr = np.frombuffer(r.content, np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is not None:
                    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception:
            if attempt < retries - 1:
                time.sleep(1)
    return None


def augment_pair(image, mask):
    """Consistent augmentation of image + mask pair."""
    if random.random() > 0.5:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)
    if random.random() > 0.5:
        image = cv2.flip(image, 0)
        mask = cv2.flip(mask, 0)
    k = random.randint(0, 3)
    if k > 0:
        image = np.rot90(image, k).copy()
        mask = np.rot90(mask, k).copy()
    if random.random() > 0.4:
        alpha = random.uniform(0.75, 1.25)
        beta = random.uniform(-20, 20)
        image = np.clip(alpha * image.astype(np.float32) + beta, 0, 255).astype(np.uint8)
    if random.random() > 0.5:
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] + random.uniform(-10, 10)) % 180
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(0.8, 1.2), 0, 255)
        image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    if random.random() > 0.7:
        noise = np.random.normal(0, random.uniform(2, 8), image.shape).astype(np.float32)
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return image, mask


def process_region(region_cfg, session, target_per_region, output_size):
    """Query OSM + download tiles for one region. Returns list of (img, mask, name)."""
    min_lat, min_lon, max_lat, max_lon, zoom, label = region_cfg

    # Query OSM for water bodies in this bbox
    elements = query_water_bodies((min_lat, min_lon, max_lat, max_lon), max_results=50)
    time.sleep(0.3)  # Rate limiting

    if not elements:
        print(f"    [{label}] No OSM elements returned, skipping.")
        return []

    print(f"    [{label}] Got {len(elements)} OSM elements")

    # Find all tiles within the bbox that have water
    x_min, y_max = lat_lon_to_tile(min_lat, min_lon, zoom)  # note y is inverted
    x_max, y_min = lat_lon_to_tile(max_lat, max_lon, zoom)

    # Ensure correct ordering
    if x_min > x_max:
        x_min, x_max = x_max, x_min
    if y_min > y_max:
        y_min, y_max = y_max, y_min

    all_tiles = [(x, y) for x in range(x_min, x_max + 1) for y in range(y_min, y_max + 1)]
    random.shuffle(all_tiles)

    results = []
    for x, y in all_tiles[:target_per_region * 3]:
        if len(results) >= target_per_region:
            break

        # Rasterize OSM polygons onto this tile
        mask = rasterize_osm_polygons(elements, x, y, zoom, tile_size=output_size)

        water_pct = (mask > 127).mean()

        # Skip tiles with very little or no water in OSM data
        if water_pct < 0.01:
            continue

        # Download the satellite tile
        img = download_tile(x, y, zoom, session)
        if img is None:
            continue

        img_resized = cv2.resize(img, (output_size, output_size), interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(mask, (output_size, output_size), interpolation=cv2.INTER_NEAREST)

        results.append((img_resized, mask_resized, f"{label}_{zoom}_{x}_{y}"))

    return results


def build_dataset(output_dir, target_total=300, aug_per_tile=3, val_split=0.15, output_size=512):
    os.makedirs(output_dir, exist_ok=True)
    for split in ["train", "val"]:
        os.makedirs(os.path.join(output_dir, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, "masks"), exist_ok=True)

    session = requests.Session()
    session.headers["User-Agent"] = "EDGE-DRIVE flood-research (educational use)"

    target_per_region = max(3, target_total // len(QUERY_REGIONS))

    print(f"Querying {len(QUERY_REGIONS)} flood-prone regions via Overpass API...")
    print(f"Target: ~{target_per_region} tiles/region, {target_total} total base tiles\n")

    all_pairs = []  # (image, mask, name)

    # Sequential processing (Overpass API rate limits)
    for region_cfg in QUERY_REGIONS:
        label = region_cfg[5]
        try:
            pairs = process_region(region_cfg, session, target_per_region, output_size)
            all_pairs.extend(pairs)
            water_pcts = [(mask > 127).mean() * 100 for _, mask, _ in pairs]
            avg_w = sum(water_pcts) / len(water_pcts) if water_pcts else 0
            print(f"  {label}: {len(pairs)} tiles (avg water: {avg_w:.1f}%)")
        except Exception as e:
            print(f"  {label}: ERROR - {e}")
        time.sleep(0.5)  # Be nice to the API

    print(f"\nCollected {len(all_pairs)} base tiles with OSM ground truth masks")

    if len(all_pairs) == 0:
        print("ERROR: No tiles collected. Check network connection.")
        return

    # Split and save
    random.shuffle(all_pairs)
    val_n = max(1, int(len(all_pairs) * val_split))

    saved_train = 0
    saved_val = 0

    for i, (img, mask, name) in enumerate(all_pairs):
        split = "val" if i < val_n else "train"

        # Save original
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(output_dir, split, "images", f"{name}.png"), img_bgr)
        cv2.imwrite(os.path.join(output_dir, split, "masks", f"{name}.png"), mask)

        if split == "val":
            saved_val += 1
            continue

        saved_train += 1

        # Augment training samples
        for aug_i in range(aug_per_tile):
            aug_img, aug_mask = augment_pair(img.copy(), mask.copy())
            aug_name = f"{name}_aug{aug_i}"
            img_bgr = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(output_dir, "train", "images", f"{aug_name}.png"), img_bgr)
            cv2.imwrite(os.path.join(output_dir, "train", "masks", f"{aug_name}.png"), aug_mask)

    train_count = len(os.listdir(os.path.join(output_dir, "train", "images")))
    val_count = len(os.listdir(os.path.join(output_dir, "val", "images")))

    info = {
        "train_samples": train_count,
        "val_samples": val_count,
        "base_tiles": len(all_pairs),
        "regions": len(QUERY_REGIONS),
        "source": "ESRI World Imagery + OpenStreetMap water polygons",
        "mask_method": "OSM water body polygon rasterization (accurate ground truth)",
        "output_size": output_size,
    }
    with open(os.path.join(output_dir, "dataset_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    print(f"\nDataset complete:")
    print(f"  Train: {train_count} images + masks")
    print(f"  Val:   {val_count} images + masks")
    print(f"  Saved to: {output_dir}")
    print(f"\nTrain with:")
    print(f"  python models/train.py --data_dir {output_dir} --epochs 50 --batch_size 4")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="data/flood_dataset")
    parser.add_argument("--samples", type=int, default=300, help="Target number of base tiles")
    parser.add_argument("--aug", type=int, default=3, help="Augmentations per training tile")
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--size", type=int, default=512, help="Output image size")
    args = parser.parse_args()

    build_dataset(args.output, target_total=args.samples, aug_per_tile=args.aug,
                  val_split=args.val_split, output_size=args.size)
