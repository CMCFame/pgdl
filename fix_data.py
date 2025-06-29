import pandas as pd
import numpy as np
import os

os.makedirs('data/processed', exist_ok=True)
np.random.seed(42)

# simulation_metrics_2283.csv
sim_metrics = pd.DataFrame({
    'scenario_id': range(1000),
    'aciertos_totales': np.random.binomial(14, 0.6, 1000),
    'premio_total': np.random.exponential(1000, 1000),
    'roi': np.random.normal(0, 0.5, 1000),
    'beneficio': np.random.normal(0, 1000, 1000)
})

sim_metrics.to_csv('data/processed/simulation_metrics_2283.csv', index=False)
print('Archivo creado exitosamente')
