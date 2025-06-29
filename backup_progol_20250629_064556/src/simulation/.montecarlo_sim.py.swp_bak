# src/simulation/montecarlo_sim.py - VERSION CORREGIDA
import pandas as pd
import numpy as np
import os
import json
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# NOTA: Reemplazamos ace_tools con implementación propia
def get_dynamic_jornada():
    """Obtener jornada de forma dinámica (reemplaza ace_tools)"""
    if 'JORNADA_ID' in os.environ:
        return os.environ['JORNADA_ID']
    
    # Buscar en archivos procesados
    processed_path = Path("data/processed")
    if processed_path.exists():
        for file in processed_path.glob("portfolio_*_*.csv"):
            parts = file.stem.split('_')
            for part in parts:
                if part.isdigit() and len(part) == 4:
                    return part
    
    return "2287"

def load_portfolio_dynamic(jornada_id):
    """Cargar portfolio de forma dinámica"""
    processed_path = Path("data/processed")
    
    # Patrones de búsqueda para portfolio
    patterns = [
        f"portfolio_grasp_{jornada_id}.csv",
        f"portfolio_preanneal_{jornada_id}.csv", 
        f"quinielas_finales_{jornada_id}.csv",
        "portfolio_latest.csv",
        "quinielas_finales.csv"
    ]
    
    for pattern in patterns:
        file_path = processed_path / pattern
        if file_path.exists():
            try:
                df = pd.read_csv(file_path)
                print(f"✅ Portfolio cargado desde: {pattern}")
                return df
            except Exception as e:
                print(f"⚠️ Error cargando {pattern}: {e}")
                continue
    
    print("⚠️ No se encontró portfolio - generando fallback")
    return create_fallback_portfolio(jornada_id)

def load_probabilities_dynamic(jornada_id):
    """Cargar probabilidades de forma dinámica"""
    processed_path = Path("data/processed")
    
    patterns = [
        f"prob_draw_adjusted_{jornada_id}.csv",
        f"probabilidades_finales_{jornada_id}.csv",
        f"match_features_{jornada_id}.csv",
        "probabilities_latest.csv"
    ]
    
    for pattern in patterns:
        file_path = processed_path / pattern
        if file_path.exists():
            try:
                df = pd.read_csv(file_path)
                # Verificar que tenga columnas de probabilidades finales
                prob_cols = ['p_final_L', 'p_final_E', 'p_final_V']
                if all(col in df.columns for col in prob_cols):
                    print(f"✅ Probabilidades cargadas desde: {pattern}")
                    return df
            except Exception as e:
                print(f"⚠️ Error cargando probabilidades {pattern}: {e}")
                continue
    
    print("⚠️ No se encontraron probabilidades - generando fallback")
    return create_fallback_probabilities()

def create_fallback_portfolio(jornada_id):
    """Crear portfolio fallback si no existe"""
    print("🔧 Generando portfolio fallback...")
    
    portfolio_data = []
    for i in range(30):  # 30 quinielas estándar
        quiniela = []
        for j in range(14):  # 14 partidos
            # Distribución realista: más locales, menos empates
            rand = np.random.random()
            if rand < 0.42:
                resultado = 'L'
            elif rand < 0.72:
                resultado = 'V'
            else:
                resultado = 'E'
            quiniela.append(resultado)
        
        row = {'quiniela_id': f'Q{i+1:02d}'}
        for j, resultado in enumerate(quiniela):
            row[f'P{j+1}'] = resultado
        
        portfolio_data.append(row)
    
    return pd.DataFrame(portfolio_data)

def create_fallback_probabilities():
    """Crear probabilidades fallback"""
    print("🔧 Generando probabilidades fallback...")
    
    prob_data = []
    for i in range(14):
        # Distribución típica de probabilidades
        p_l = np.random.uniform(0.25, 0.55)
        p_v = np.random.uniform(0.25, 0.55) 
        p_e = 1.0 - p_l - p_v
        
        # Asegurar que p_e sea razonable
        if p_e < 0.15:
            total = p_l + p_v + 0.15
            p_l = p_l / total * 0.85
            p_v = p_v / total * 0.85
            p_e = 0.15
        
        prob_data.append({
            'match_no': i + 1,
            'p_final_L': p_l,
            'p_final_E': p_e,
            'p_final_V': p_v
        })
    
    return pd.DataFrame(prob_data)

def calcular_probabilidad_acierto_quiniela(quiniela, prob_df):
    """Calcular vector de probabilidades de acierto para una quiniela"""
    prob_acierto = []
    
    for i, signo in enumerate(quiniela):
        if i < len(prob_df):
            prob = prob_df.iloc[i].get(f'p_final_{signo}', 0.33)
        else:
            prob = 0.33  # Fallback realista
        
        prob_acierto.append(max(0.01, min(0.99, prob)))  # Clamp entre 1% y 99%
    
    return np.array(prob_acierto)

def simulacion_monte_carlo_avanzada(portfolio_df, prob_df, n_simulaciones=10000):
    """Simulación Monte Carlo avanzada para evaluar portfolio"""
    
    print(f"🎲 Ejecutando simulación Monte Carlo ({n_simulaciones:,} iteraciones)...")
    
    resultados = {}
    
    # Extraer quinielas del portfolio
    quinielas = []
    quiniela_ids = []
    
    for _, row in portfolio_df.iterrows():
        quiniela = [row[f'P{i+1}'] for i in range(14)]
        quinielas.append(quiniela)
        quiniela_ids.append(row.get('quiniela_id', f'Q{len(quinielas)}'))
    
    print(f"📊 Simulando {len(quinielas)} quinielas...")
    
    # Simular cada quiniela
    for i, (quiniela, qid) in enumerate(zip(quinielas, quiniela_ids)):
        # Calcular probabilidades de acierto
        prob_acierto = calcular_probabilidad_acierto_quiniela(quiniela, prob_df)
        
        # Simulación vectorizada
        aciertos = np.random.binomial(1, prob_acierto, size=(n_simulaciones, 14))
        hits = aciertos.sum(axis=1)
        
        # Calcular métricas
        mu_hits = hits.mean()
        sigma_hits = hits.std()
        pr_11 = (hits >= 11).mean()
        pr_10 = (hits >= 10).mean()
        pr_9 = (hits >= 9).mean()
        
        # ROI estimado (asumiendo premios típicos)
        # Distribución de premios aproximada para Progol
        roi_sim = []
        for n_hits in hits:
            if n_hits >= 11:
                premio = 50000  # Premio típico ≥11 aciertos
            elif n_hits >= 10:
                premio = 1000   # Premio típico 10 aciertos
            elif n_hits >= 9:
                premio = 100    # Premio típico 9 aciertos
            else:
                premio = 0
            
            costo = 10  # Costo típico por quiniela
            roi = (premio - costo) / costo if costo > 0 else 0
            roi_sim.append(roi)
        
        roi_esperado = np.mean(roi_sim)
        
        resultados[qid] = {
            'mu_hits': mu_hits,
            'sigma_hits': sigma_hits,
            'pr_11': pr_11,
            'pr_10': pr_10,
            'pr_9': pr_9,
            'roi_esperado': roi_esperado,
            'prob_acierto_promedio': prob_acierto.mean(),
            'quiniela': ''.join(quiniela)
        }
        
        if (i + 1) % 10 == 0:
            print(f"  Simuladas {i+1}/{len(quinielas)} quinielas...")
    
    # Calcular métricas del portfolio completo
    todas_pr11 = [r['pr_11'] for r in resultados.values()]
    portfolio_pr11 = 1 - np.prod([1 - pr11 for pr11 in todas_pr11])  # Probabilidad de al menos una ≥11
    
    media_pr11 = np.mean(todas_pr11)
    mejor_quiniela = max(resultados.keys(), key=lambda k: resultados[k]['pr_11'])
    mejor_pr11 = resultados[mejor_quiniela]['pr_11']
    
    media_mu_hits = np.mean([r['mu_hits'] for r in resultados.values()])
    media_roi = np.mean([r['roi_esperado'] for r in resultados.values()])
    
    # Crear resumen
    resumen = {
        'portfolio_pr11': portfolio_pr11,
        'media_pr11': media_pr11,
        'mejor_quiniela': mejor_quiniela,
        'mejor_pr11': mejor_pr11,
        'media_mu_hits': media_mu_hits,
        'media_roi': media_roi,
        'n_quinielas': len(quinielas),
        'n_simulaciones': n_simulaciones,
        'prob_tipica_acierto': np.mean([r['prob_acierto_promedio'] for r in resultados.values()])
    }
    
    print(f"✅ Simulación completada:")
    print(f"  📊 Pr[≥11] Promedio: {media_pr11:.3%}")
    print(f"  🎯 Mejor Quiniela: {mejor_quiniela} (Pr[≥11]={mejor_pr11:.3%})")
    print(f"  🔢 μ Hits Promedio: {media_mu_hits:.2f}")
    print(f"  💰 ROI Estimado: {media_roi:.1%}")
    
    return resultados, resumen

def generar_diagnostico_calidad(prob_df, resumen):
    """Generar diagnóstico de calidad de probabilidades"""
    
    print(f"\n🔍 Diagnóstico de Calidad:")
    
    # Analizar distribución de probabilidades
    prob_cols = ['p_final_L', 'p_final_E', 'p_final_V']
    if all(col in prob_df.columns for col in prob_cols):
        prob_max = prob_df[prob_cols].max(axis=1)
        prob_min = prob_df[prob_cols].min(axis=1)
        
        print(f"📊 Calidad de probabilidades: Prob. máxima promedio={prob_max.mean():.1%}, Rango=[{prob_max.min():.1%}, {prob_max.max():.1%}]")
        
        # Diagnosticar si Pr[≥11] es muy bajo
        if resumen['media_pr11'] < 0.02:
            print(f"🔍 Diagnóstico de Resultados")
            print(f"❌ Pr[≥11] muy bajo - Posibles causas:")
            print(f"• Los odds originales pueden ser demasiado altos")
            print(f"• Las probabilidades generadas son demasiado conservadoras") 
            print(f"• Verifica que los odds estén en formato decimal correcto")
        
        # Debug probabilidades originales
        print(f"🔍 Debug: Probabilidades Originales")
        print(f"🎯 Probabilidad típica de acierto: {resumen['prob_tipica_acierto']:.1%}")
        aciertos_esperados = resumen['prob_tipica_acierto'] * 14
        print(f"📊 Aciertos esperados teóricos: {aciertos_esperados:.1f}")

def exportar_resultados_simulacion(resultados, resumen, jornada_id):
    """Exportar resultados de simulación"""
    
    processed_path = Path("data/processed")
    processed_path.mkdir(parents=True, exist_ok=True)
    
    # Exportar resultados detallados
    df_resultados = pd.DataFrame.from_dict(resultados, orient='index')
    df_resultados.reset_index(inplace=True)
    df_resultados.rename(columns={'index': 'quiniela_id'}, inplace=True)
    
    results_file = processed_path / f"simulation_results_{jornada_id}.csv"
    df_resultados.to_csv(results_file, index=False)
    
    # Exportar resumen
    summary_file = processed_path / f"simulation_summary_{jornada_id}.json"
    with open(summary_file, 'w') as f:
        json.dump(resumen, f, indent=2, default=str)
    
    print(f"✅ Resultados exportados:")
    print(f"  📁 Detallados: {results_file}")
    print(f"  📁 Resumen: {summary_file}")
    
    return results_file, summary_file

def generar_recomendaciones(resumen):
    """Generar recomendaciones para mejorar el sistema"""
    
    print(f"\n💡 Recomendaciones para Mejorar:")
    
    if resumen['media_pr11'] < 0.05:
        print(f"🔧 Verificar formato de odds: Asegúrate de que estén en formato decimal (ej: 2.5, no 5/2)")
        print(f"📊 Revisar datos de entrada: Verifica que los odds sean realistas para fútbol")
        print(f"⚙️ Ajustar parámetros: Reduce el ajuste bayesiano conservador")
    
    if resumen['media_roi'] < -0.3:
        print(f"🎯 Modelo más agresivo: Considera usar estrategias de optimización más arriesgadas")
    
    print(f"📈 Análisis histórico: Compara con resultados reales de jornadas pasadas")

def main():
    """Función principal del módulo de simulación"""
    
    try:
        # Obtener jornada dinámica
        jornada_id = get_dynamic_jornada()
        print(f"🎯 Ejecutando simulación Monte Carlo para jornada: {jornada_id}")
        
        # Cargar datos
        portfolio_df = load_portfolio_dynamic(jornada_id)
        prob_df = load_probabilities_dynamic(jornada_id)
        
        # Ejecutar simulación
        resultados, resumen = simulacion_monte_carlo_avanzada(portfolio_df, prob_df)
        
        # Generar diagnósticos
        generar_diagnostico_calidad(prob_df, resumen)
        
        # Exportar resultados
        exportar_resultados_simulacion(resultados, resumen, jornada_id)
        
        # Generar recomendaciones
        generar_recomendaciones(resumen)
        
        print(f"\n✅ Simulación Monte Carlo completada para jornada {jornada_id}")
        
        return resultados, resumen
        
    except Exception as e:
        print(f"❌ Error en simulación Monte Carlo: {e}")
        raise

if __name__ == "__main__":
    main()