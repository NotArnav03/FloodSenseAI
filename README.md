# Edge-Drive: Flood Risk Intelligence System

An AI-powered disaster intelligence platform that analyzes flood probability maps to identify high-risk zones and optimize emergency resource allocation.

## Features

- **Real Flood Data Processing**: Supports 446+ flood probability maps across 11 regions (India, USA, Nigeria, Pakistan, etc.)
- **Risk Analysis Engine**: Combines flood probability with population density to compute composite risk scores
- **Smart Resource Allocation**: Proportionally distributes relief resources (food, medical kits, boats) to highest-risk zones
- **Multi-Interface Support**:
  - Command-line interface for batch processing
  - Streamlit web dashboard for interactive analysis
- **Batch Processing**: Analyze multiple maps by region with summary statistics

## Installation

```bash
# Clone the repository
git clone https://github.com/NotArnav03/edge-drive.git
cd edge-drive

# Install dependencies
pip install -r requirements.txt
```

## Data Setup

Place flood probability map files (`.npy` format, 512x512 float32) in the `data/` directory.

Expected format: `{Region}_{ID}_flood_prob.npy` (e.g., `India_1017769_flood_prob.npy`)

## Usage

### Command Line Interface

```bash
# List all available flood maps
python main.py --list

# Process a specific map
python main.py --map "India_1017769_flood_prob.npy"

# Batch process maps from a region
python main.py --region India --top 5

# Process without visualization
python main.py --map "USA_123456_flood_prob.npy" --no-viz

# Save visualizations to folder
python main.py --region Pakistan --top 3 --save-dir output/
```

### Web Dashboard

```bash
cd app
streamlit run app.py
```

## Project Structure

```
edge-drive/
├── main.py                 # CLI entry point
├── requirements.txt        # Python dependencies
├── app/
│   └── app.py              # Streamlit web dashboard
├── core/
│   ├── risk_engine.py      # Flood risk computation engine
│   └── allocation_engine.py # Resource allocation logic
├── data/                   # Flood probability maps (not in repo)
└── tests/
    └── test_risk_engine.py # Unit tests
```

## Core Components

### RiskEngine
- Loads flood probability maps from `.npy` files
- Generates simulated population density grids
- Computes weighted risk maps: `risk = 0.6 * flood + 0.4 * population * flood`
- Identifies top-k highest risk zones

### AllocationEngine
- Distributes resources proportionally to zone risk scores
- Supports cluster-based allocation for practical deployment

## Output Example

```
Flood mask loaded: India_1017769_flood_prob.npy
  Shape: (512, 512)
  Min: 0.0003, Max: 0.9852
Risk map computed. Max risk: 0.8016

Risk Statistics:
  High risk cells (>0.5): 93,427
  Critical cells (>0.8): 1

Top 10 High-Risk Zones:
----------------------------------------
  1. Zone (298, 163) | Risk: 0.8016
  2. Zone (291, 169) | Risk: 0.7937
  ...

Resource Allocation (Top 5 Zones):
------------------------------------------------------------
Zone                Risk     Food  Medical  Boats
------------------------------------------------------------
(298, 163)        0.8016      254       60      2
(291, 169)        0.7937      251       60      2
```

## Supported Regions

- Bolivia
- Ghana
- India
- Mekong
- Nigeria
- Pakistan
- Paraguay
- Somalia
- Spain
- Sri-Lanka
- USA

## Requirements

- Python 3.8+
- NumPy >= 1.20.0
- Matplotlib >= 3.5.0
- SciPy >= 1.7.0
- Streamlit >= 1.20.0 (for web interface)

## License

MIT License
