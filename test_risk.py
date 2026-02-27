from core.risk_engine import RiskEngine

engine = RiskEngine()
engine.load_flood_mask()
engine.generate_population_grid()
engine.compute_risk()

top_zones = engine.get_top_risk_zones()
print("Top high-risk zones:", top_zones)