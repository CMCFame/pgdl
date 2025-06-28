# Crear un script que genere TODOS los archivos posibles
cat > create_all_data.py << 'EOF'
import pandas as pd
import numpy as np
import os

def crear_todos_los_archivos():
    # Crear directorios
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('data/raw', exist_ok=True)
    
    np.random.seed(42)  # Reproducibilidad
    
    # Lista de archivos que puede buscar el dashboard
    archivos_comunes = [
        'portfolio_final_2283.csv',
        'prob_draw_adjusted_2283.csv', 
        'probabilidades_2283.csv',
        'simulacion_montecarlo_2283.csv',
        'match_features_2283.csv',
        'core_quinielas_2283.csv',
        'satellite_quinielas_2283.csv',
        'metricas_portfolio_2283.csv'
    ]
    
    # 1. Portfolio final
    portfolio = pd.DataFrame({
        'quiniela_id': [f'Q{i:02d}' for i in range(1, 31)],
        'tipo': ['Core']*4 + ['Satelite']*26,
        'esperanza_aciertos': np.random.uniform(10.5, 13.2, 30),
        'prob_11_plus': np.random.uniform(0.08, 0.15, 30),
        'roi_esperado': np.random.uniform(-0.2, 0.8, 30),
        'costo': [15]*30,
        'premio_esperado': np.random.uniform(0, 200, 30)
    })
    
    # Agregar columnas P1-P14 (partidos)
    for i in range(1, 15):
        portfolio[f'P{i}'] = np.random.choice(['L', 'E', 'V'], 30, p=[0.45, 0.25, 0.3])
    
    portfolio.to_csv('data/processed/portfolio_final_2283.csv', index=False)
    
    # 2. Probabilidades ajustadas con draw propensity
    prob_draw = pd.DataFrame({
        'match_id': [f'2283-{i}' for i in range(1, 15)],
        'partido_num': range(1, 15),
        'equipo_local': [f'Local_{i}' for i in range(1, 15)],
        'equipo_visitante': [f'Visitante_{i}' for i in range(1, 15)],
        'p_final_L': np.random.uniform(0.25, 0.65, 14),
        'p_final_E': np.random.uniform(0.15, 0.35, 14),
        'p_final_V': np.random.uniform(0.25, 0.65, 14),
        'draw_propensity_flag': np.random.choice([True, False], 14, p=[0.3, 0.7]),
        'etiqueta': np.random.choice(['Normal', 'Ancla', 'TendenciaX'], 14),
        'signo_argmax': np.random.choice(['L', 'E', 'V'], 14)
    })
    
    # Normalizar probabilidades
    for i in range(len(prob_draw)):
        total = prob_draw.loc[i, ['p_final_L', 'p_final_E', 'p_final_V']].sum()
        prob_draw.loc[i, ['p_final_L', 'p_final_E', 'p_final_V']] /= total
    
    prob_draw.to_csv('data/processed/prob_draw_adjusted_2283.csv', index=False)
    
    # 3. Simulación Monte Carlo
    simulacion = pd.DataFrame({
        'scenario_id': range(1000),
        'aciertos_totales': np.random.binomial(14, 0.6, 1000),
        'premio_total': np.random.exponential(1000, 1000),
        'roi': np.random.normal(0, 0.5, 1000),
        'beneficio': np.random.normal(0, 1000, 1000)
    })
    simulacion.to_csv('data/processed/simulacion_montecarlo_2283.csv', index=False)
    
    print("✅ Archivos creados:")
    for archivo in os.listdir('data/processed'):
        print(f"  📄 {archivo}")

if __name__ == "__main__":
    crear_todos_los_archivos()
EOF
