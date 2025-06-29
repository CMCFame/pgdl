import pandas as pd
import numpy as np
import os

os.makedirs('data/processed', exist_ok=True)

# Crear prob_draw_adjusted_2283.csv
prob_draw = pd.DataFrame({
    'match_id': [f'2283-{i}' for i in range(1, 15)],
    'p_final_L': [0.4] * 14,
    'p_final_E': [0.25] * 14, 
    'p_final_V': [0.35] * 14,
    'draw_propensity_flag': [False] * 14,
    'etiqueta': ['Normal'] * 14
})

prob_draw.to_csv('data/processed/prob_draw_adjusted_2283.csv', index=False)
print('✅ Archivo creado!')
