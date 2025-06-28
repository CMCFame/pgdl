#!/usr/bin/env python3
"""
Script para crear datos de ejemplo para el dashboard de Progol Engine
Esto permite ver el dashboard funcionando sin necesidad de datos reales
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import json

def crear_directorios():
    """Crear directorios necesarios"""
    directories = [
        'data/processed',
        'data/raw', 
        'data/dashboard',
        'data/json_previas'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Directorio creado: {directory}")

def crear_portfolio_final():
    """Crear archivo portfolio_final_2283.csv"""
    print("📊 Creando portfolio_final_2283.csv...")
    
    # Crear 30 quinielas de ejemplo
    np.random.seed(42)  # Para reproducibilidad
    
    quinielas = []
    for i in range(30):
        quiniela = {
            'quiniela_id': f'Q{i+1:02d}',
            'tipo': 'Core' if i < 4 else 'Satelite',
            'esperanza_aciertos': np.random.uniform(10.5, 13.2),
            'prob_11_plus': np.random.uniform(0.08, 0.15),
            'roi_esperado': np.random.uniform(-0.2, 0.8),
            'costo': 15,
            'premio_esperado': np.random.uniform(0, 200)
        }
        
        # Agregar 14 partidos (P1-P14)
        for j in range(1, 15):
            resultado = np.random.choice(['L', 'E', 'V'], p=[0.4, 0.25, 0.35])
            quiniela[f'P{j}'] = resultado
        
        quinielas.append(quiniela)
    
    df_portfolio = pd.DataFrame(quinielas)
    df_portfolio.to_csv('data/processed/portfolio_final_2283.csv', index=False)
    print(f"✅ Creado portfolio con {len(df_portfolio)} quinielas")
    return df_portfolio

def crear_probabilidades():
    """Crear archivo de probabilidades por partido"""
    print("🎯 Creando probabilidades_2283.csv...")
    
    partidos = []
    for i in range(1, 15):
        # Generar probabilidades que sumen 1
        probs = np.random.dirichlet([2, 1, 2])  # Favorece L y V sobre E
        
        partido = {
            'match_id': f'2283-{i}',
            'partido_num': i,
            'equipo_local': f'Equipo_L_{i}',
            'equipo_visitante': f'Equipo_V_{i}',
            'p_final_L': probs[0],
            'p_final_E': probs[1], 
            'p_final_V': probs[2],
            'p_raw_L': probs[0] * np.random.uniform(0.9, 1.1),
            'p_raw_E': probs[1] * np.random.uniform(0.9, 1.1),
            'p_raw_V': probs[2] * np.random.uniform(0.9, 1.1),
            'lambda_local': np.random.uniform(0.8, 2.5),
            'lambda_visitante': np.random.uniform(0.8, 2.5),
            'etiqueta': np.random.choice(['Normal', 'Ancla', 'TendenciaX'], p=[0.7, 0.2, 0.1])
        }
        partidos.append(partido)
    
    df_probs = pd.DataFrame(partidos)
    df_probs.to_csv('data/processed/probabilidades_2283.csv', index=False)
    print(f"✅ Creado {len(df_probs)} partidos con probabilidades")
    return df_probs

def crear_simulacion():
    """Crear resultados de simulación Monte Carlo"""
    print("🎲 Creando simulacion_montecarlo_2283.csv...")
    
    # Simular resultados de 1000 scenarios
    scenarios = []
    for i in range(1000):
        scenario = {
            'scenario_id': i,
            'aciertos_totales': np.random.binomial(14, 0.6),  # ~8-9 aciertos promedio
            'premio_total': 0,
            'roi': 0,
            'beneficio': 0
        }
        
        # Calcular premios basados en aciertos
        if scenario['aciertos_totales'] >= 14:
            scenario['premio_total'] = 500000  # Premio categoría 1
        elif scenario['aciertos_totales'] >= 13:
            scenario['premio_total'] = 50000   # Premio categoría 2
        elif scenario['aciertos_totales'] >= 12:
            scenario['premio_total'] = 5000    # Premio categoría 3
        elif scenario['aciertos_totales'] >= 11:
            scenario['premio_total'] = 500     # Premio categoría 4
        
        costo_total = 30 * 15  # 30 quinielas x $15
        scenario['beneficio'] = scenario['premio_total'] - costo_total
        scenario['roi'] = scenario['beneficio'] / costo_total if costo_total > 0 else 0
        
        scenarios.append(scenario)
    
    df_sim = pd.DataFrame(scenarios)
    df_sim.to_csv('data/processed/simulacion_montecarlo_2283.csv', index=False)
    print(f"✅ Creada simulación con {len(df_sim)} escenarios")
    return df_sim

def crear_datos_raw():
    """Crear algunos datos raw de ejemplo"""
    print("📁 Creando datos raw de ejemplo...")
    
    # Crear progol.csv básico
    progol_data = []
    for i in range(1, 15):
        match = {
            'concurso_id': 2283,
            'fecha': '2024-12-15',
            'match_no': i,
            'liga': 'Liga MX',
            'home': f'Equipo_Local_{i}',
            'away': f'Equipo_Visitante_{i}',
            'l_g': np.random.randint(0, 4),
            'a_g': np.random.randint(0, 4),
            'resultado': np.random.choice(['L', 'E', 'V']),
            'premio_1': 0,
            'premio_2': 0
        }
        progol_data.append(match)
    
    df_progol = pd.DataFrame(progol_data)
    df_progol.to_csv('data/raw/progol_sample.csv', index=False)
    
    # Crear odds.csv básico
    odds_data = []
    for i in range(1, 15):
        odds = {
            'home': f'Equipo_Local_{i}',
            'away': f'Equipo_Visitante_{i}',
            'fecha': '2024-12-15',
            'odds_L': np.random.uniform(1.5, 4.0),
            'odds_E': np.random.uniform(2.8, 3.8),
            'odds_V': np.random.uniform(1.8, 5.0)
        }
        odds_data.append(odds)
    
    df_odds = pd.DataFrame(odds_data)
    df_odds.to_csv('data/raw/odds_sample.csv', index=False)
    
    print("✅ Datos raw de ejemplo creados")

def crear_configuracion():
    """Crear archivo .env con configuración básica"""
    if not os.path.exists('.env'):
        print("⚙️ Creando archivo .env...")
        
        env_content = """# === CONFIGURACIÓN PROGOL ENGINE ===
JORNADA_ID=2283
N_QUINIELAS=30
COSTO_BOLETO=15
PREMIO_CAT2=90000
N_MONTECARLO_SAMPLES=1000

# === ARCHIVOS DEFAULT ===
PROGOL_CSV=data/raw/progol_sample.csv
ODDS_CSV=data/raw/odds_sample.csv
"""
        
        with open('.env', 'w') as f:
            f.write(env_content)
        print("✅ Archivo .env creado")

def generar_resumen():
    """Mostrar resumen de archivos creados"""
    print("\n" + "="*50)
    print("📊 RESUMEN DE ARCHIVOS CREADOS")
    print("="*50)
    
    archivos = [
        'data/processed/portfolio_final_2283.csv',
        'data/processed/probabilidades_2283.csv', 
        'data/processed/simulacion_montecarlo_2283.csv',
        'data/raw/progol_sample.csv',
        'data/raw/odds_sample.csv',
        '.env'
    ]
    
    for archivo in archivos:
        if os.path.exists(archivo):
            size = os.path.getsize(archivo)
            print(f"✅ {archivo} ({size:,} bytes)")
        else:
            print(f"❌ {archivo}")
    
    print("\n🎯 PRÓXIMOS PASOS:")
    print("1. El dashboard ya debería funcionar")
    print("2. Ejecuta: streamlit run streamlit_app/dashboard.py")
    print("3. Ve a: http://localhost:8501")
    print("\n🎉 ¡Datos de ejemplo listos!")

def main():
    """Función principal"""
    print("🚀 CREANDO DATOS DE EJEMPLO PARA PROGOL ENGINE")
    print("="*50)
    
    try:
        crear_directorios()
        crear_portfolio_final()
        crear_probabilidades()
        crear_simulacion()
        crear_datos_raw()
        crear_configuracion()
        generar_resumen()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    main()
