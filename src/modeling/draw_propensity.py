#!/usr/bin/env python3	
"""
draw_propensity.py - Regla de empates robusta
Implementa la regla de draw propensity de la metodología
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Configurar logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("draw_propensity")

def detectar_jornada():
    """Detectar jornada desde archivos disponibles"""
    jornada = None
    
    # Buscar en data/processed
    processed_dir = Path("data/processed")
    if processed_dir.exists():
        for archivo in processed_dir.glob("*.csv"):
            try:
                parts = archivo.stem.split('_')
                for part in parts:
                    if part.isdigit() and len(part) >= 4:
                        jornada = int(part)
                        break
                if jornada:
                    break
            except:
                continue
    
    return jornada or 2287

def cargar_probabilidades_finales():
    """Cargar probabilidades finales desde bayesian adjustment"""
    jornada = detectar_jornada()
    
    paths_to_try = [
        f"data/processed/prob_final_{jornada}.csv",
        "data/processed/prob_final.csv",
        f"data/processed/prob_blend_{jornada}.csv",
        "data/processed/prob_blend.csv"
    ]
    
    for path in paths_to_try:
        if Path(path).exists():
            try:
                df = pd.read_csv(path)
                
                # Verificar columnas necesarias
                required_cols = ['p_final_L', 'p_final_E', 'p_final_V']
                alt_cols = ['p_blend_L', 'p_blend_E', 'p_blend_V']
                
                if all(col in df.columns for col in required_cols):
                    logger.info(f"Probabilidades finales cargadas desde: {path}")
                    return df
                elif all(col in df.columns for col in alt_cols):
                    # Renombrar columnas
                    df = df.rename(columns={
                        'p_blend_L': 'p_final_L',
                        'p_blend_E': 'p_final_E',
                        'p_blend_V': 'p_final_V'
                    })
                    logger.info(f"Probabilidades blend renombradas desde: {path}")
                    return df
                else:
                    logger.warning(f"Archivo {path} no tiene columnas correctas")
                    
            except Exception as e:
                logger.warning(f"Error cargando {path}: {e}")
                continue
    
    logger.warning("No se encontraron probabilidades. Generando por defecto.")
    return generar_probabilidades_default()

def generar_probabilidades_default():
    """Generar probabilidades por defecto"""
    jornada = detectar_jornada()
    
    df_prob = pd.DataFrame({
        'match_id': [f"{jornada}-{i+1}" for i in range(14)],
        'p_final_L': [0.40, 0.35, 0.50, 0.30, 0.45, 0.38, 0.42, 0.33, 0.48, 0.37, 0.41, 0.36, 0.44, 0.39],
        'p_final_E': [0.30, 0.32, 0.28, 0.35, 0.29, 0.31, 0.30, 0.34, 0.27, 0.33, 0.30, 0.32, 0.28, 0.31],
        'p_final_V': [0.30, 0.33, 0.22, 0.35, 0.26, 0.31, 0.28, 0.33, 0.25, 0.30, 0.29, 0.32, 0.28, 0.30]
    })
    
    logger.info("Probabilidades por defecto generadas")
    return df_prob

def aplicar_regla_draw_propensity(df_prob, umbral_diferencia=0.08, boost_empate=0.06):
    """
    Aplicar regla de draw propensity según la metodología:
    
    Si abs(p_L - p_V) < umbral_diferencia AND p_E > max(p_L, p_V):
        p_E += boost_empate
        p_L -= boost_empate/2
        p_V -= boost_empate/2
    
    Luego normalizar para que sumen 1
    """
    logger.info("Aplicando regla de draw propensity...")
    
    df_adjusted = df_prob.copy()
    partidos_ajustados = 0
    
    for i, row in df_adjusted.iterrows():
        p_l = row['p_final_L']
        p_e = row['p_final_E']
        p_v = row['p_final_V']
        
        # Condiciones de la regla
        diferencia_lv = abs(p_l - p_v)
        empate_mayor = p_e > max(p_l, p_v)
        
        if diferencia_lv < umbral_diferencia and empate_mayor:
            # Aplicar ajuste
            p_e_new = p_e + boost_empate
            p_l_new = p_l - boost_empate/2
            p_v_new = p_v - boost_empate/2
            
            # Asegurar que no sean negativos
            p_l_new = max(0.01, p_l_new)
            p_v_new = max(0.01, p_v_new)
            p_e_new = max(0.01, p_e_new)
            
            # Normalizar para que sumen 1
            total = p_l_new + p_e_new + p_v_new
            
            df_adjusted.loc[i, 'p_final_L'] = p_l_new / total
            df_adjusted.loc[i, 'p_final_E'] = p_e_new / total
            df_adjusted.loc[i, 'p_final_V'] = p_v_new / total
            
            partidos_ajustados += 1
            
            logger.info(f"Partido {i+1}: L={p_l:.3f}→{p_l_new/total:.3f}, "
                       f"E={p_e:.3f}→{p_e_new/total:.3f}, "
                       f"V={p_v:.3f}→{p_v_new/total:.3f}")
    
    logger.info(f"Regla aplicada a {partidos_ajustados} partidos de {len(df_adjusted)}")
    return df_adjusted

def verificar_probabilidades_validas(df):
    """Verificar que las probabilidades sean válidas después del ajuste"""
    prob_cols = ['p_final_L', 'p_final_E', 'p_final_V']
    
    # Verificar rango [0,1]
    for col in prob_cols:
        if (df[col] < 0).any():
            logger.warning(f"Probabilidades negativas detectadas en {col}")
            df[col] = df[col].clip(lower=0.001)
        
        if (df[col] > 1).any():
            logger.warning(f"Probabilidades > 1 detectadas en {col}")
            df[col] = df[col].clip(upper=0.999)
    
    # Verificar que sumen 1
    prob_sum = df[prob_cols].sum(axis=1)
    tolerance = 0.01
    
    if not np.allclose(prob_sum, 1.0, atol=tolerance):
        logger.warning("Probabilidades no suman 1. Renormalizando...")
        for col in prob_cols:
            df[col] = df[col] / prob_sum
    
    return df

def proyectar_rangos_globales(df_adjusted, rangos_target=None):
    """
    Proyectar al simplex si violan rangos L-E-V totales por cartilla
    Según metodología: 35-41% L, 25-33% E, 30-36% V
    """
    if rangos_target is None:
        rangos_target = {
            'L': (0.35, 0.41),
            'E': (0.25, 0.33),
            'V': (0.30, 0.36)
        }
    
    logger.info("Verificando rangos globales...")
    
    # Calcular distribución actual
    n_partidos = len(df_adjusted)
    total_l = df_adjusted['p_final_L'].sum()
    total_e = df_adjusted['p_final_E'].sum()
    total_v = df_adjusted['p_final_V'].sum()
    
    pct_l = total_l / n_partidos
    pct_e = total_e / n_partidos
    pct_v = total_v / n_partidos
    
    logger.info(f"Distribución actual: L={pct_l:.3f}, E={pct_e:.3f}, V={pct_v:.3f}")
    
    # Verificar si necesita ajuste
    ajuste_necesario = (
        not (rangos_target['L'][0] <= pct_l <= rangos_target['L'][1]) or
        not (rangos_target['E'][0] <= pct_e <= rangos_target['E'][1]) or
        not (rangos_target['V'][0] <= pct_v <= rangos_target['V'][1])
    )
    
    if not ajuste_necesario:
        logger.info("Rangos globales dentro de límites. No se requiere ajuste.")
        return df_adjusted
    
    logger.info("Aplicando proyección a rangos globales...")
    
    # Targets centrales de cada rango
    target_l = np.mean(rangos_target['L'])
    target_e = np.mean(rangos_target['E'])
    target_v = np.mean(rangos_target['V'])
    
    # Factores de ajuste
    factor_l = target_l / pct_l if pct_l > 0 else 1.0
    factor_e = target_e / pct_e if pct_e > 0 else 1.0
    factor_v = target_v / pct_v if pct_v > 0 else 1.0
    
    # Aplicar ajuste suave
    df_projected = df_adjusted.copy()
    
    # Ajuste ponderado (50% hacia target, 50% original)
    alpha = 0.5
    
    df_projected['p_final_L'] = df_adjusted['p_final_L'] * (1 + alpha * (factor_l - 1))
    df_projected['p_final_E'] = df_adjusted['p_final_E'] * (1 + alpha * (factor_e - 1))
    df_projected['p_final_V'] = df_adjusted['p_final_V'] * (1 + alpha * (factor_v - 1))
    
    # Renormalizar
    prob_cols = ['p_final_L', 'p_final_E', 'p_final_V']
    prob_sum = df_projected[prob_cols].sum(axis=1)
    
    for col in prob_cols:
        df_projected[col] = df_projected[col] / prob_sum
    
    # Verificar nuevo resultado
    new_total_l = df_projected['p_final_L'].sum()
    new_total_e = df_projected['p_final_E'].sum()
    new_total_v = df_projected['p_final_V'].sum()
    
    new_pct_l = new_total_l / n_partidos
    new_pct_e = new_total_e / n_partidos
    new_pct_v = new_total_v / n_partidos
    
    logger.info(f"Distribución ajustada: L={new_pct_l:.3f}, E={new_pct_e:.3f}, V={new_pct_v:.3f}")
    
    return df_projected

def marcar_draw_propensity_flags(df_final):
    """
    Marcar partidos con draw propensity - VERSIÓN MÁS SENSIBLE
    
    Criterios ajustados para detectar más casos de empate probable:
    1. Empate relativamente alto (> 28% en lugar de 32%)
    2. O empate competitivo (diferencia con máximo < 15%)
    3. O partidos muy cerrados (diferencia L-V < 12%)
    """
    df_flags = df_final.copy()
    
    # Calcular métricas auxiliares
    max_prob = df_flags[['p_final_L', 'p_final_E', 'p_final_V']].max(axis=1)
    diff_to_empate = max_prob - df_flags['p_final_E']
    diff_lv = abs(df_flags['p_final_L'] - df_flags['p_final_V'])
    
    # CRITERIOS MÁS SENSIBLES:
    
    # Criterio 1: Empate alto (>28%)
    criterio_empate_alto = df_flags['p_final_E'] > 0.28
    
    # Criterio 2: Empate competitivo (diferencia con máximo < 15%)
    criterio_empate_competitivo = diff_to_empate < 0.15
    
    # Criterio 3: Partidos muy cerrados (L vs V < 12%)
    criterio_cerrado = diff_lv < 0.12
    
    # Criterio 4: Empate es top-2 resultado más probable
    sorted_probs = df_flags[['p_final_L', 'p_final_E', 'p_final_V']].apply(
        lambda row: sorted(row, reverse=True), axis=1
    )
    criterio_top2 = df_flags['p_final_E'] >= sorted_probs.apply(lambda x: x[1])
    
    # COMBINACIÓN: cualquiera de los criterios
    df_flags['draw_propensity_flag'] = (
        criterio_empate_alto | 
        (criterio_empate_competitivo & criterio_top2) |
        (criterio_cerrado & (df_flags['p_final_E'] > 0.25))
    )
    
    n_flags = df_flags['draw_propensity_flag'].sum()
    
    # Log de diagnóstico
    print(f"=== DIAGNÓSTICO DRAW PROPENSITY ===")
    print(f"Partidos con empate > 28%: {criterio_empate_alto.sum()}")
    print(f"Partidos con empate competitivo: {criterio_empate_competitivo.sum()}")
    print(f"Partidos cerrados (L-V < 12%): {criterio_cerrado.sum()}")
    print(f"Partidos con empate top-2: {criterio_top2.sum()}")
    print(f"TOTAL marcados con draw_propensity_flag: {n_flags}")
    
    if n_flags > 0:
        print("\nPartidos marcados:")
        marcados = df_flags[df_flags['draw_propensity_flag']]
        for idx, row in marcados.iterrows():
            print(f"Partido {row.get('partido', idx+1)}: "
                  f"L={row['p_final_L']:.3f}, E={row['p_final_E']:.3f}, V={row['p_final_V']:.3f}")
    
    return df_flags


def guardar_prob_draw(df_final, output_path=None):
    """Guardar probabilidades finales ajustadas por draw propensity"""
    if output_path is None:
        jornada = detectar_jornada()
        output_path = f"data/processed/prob_draw_adjusted_{jornada}.csv"
    
    # Crear directorio si no existe
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Guardar
    df_final.to_csv(output_path, index=False)
    logger.info(f"Probabilidades draw-adjusted guardadas en: {output_path}")
    
    # También guardar copias genéricas
    generic_path = "data/processed/prob_draw_adjusted.csv"
    df_final.to_csv(generic_path, index=False)
    
    # Mantener también prob_final.csv actualizado
    final_path = "data/processed/prob_final.csv"
    df_final.drop(columns=['draw_propensity_flag'], errors='ignore').to_csv(final_path, index=False)
    
    logger.info(f"Copias guardadas en: {generic_path} y {final_path}")

def pipeline_draw_propensity_completo():
    """Pipeline completo de draw propensity"""
    try:
        logger.info("=== INICIANDO PIPELINE DRAW PROPENSITY ===")
        
        # 1. Cargar probabilidades finales
        df_prob = cargar_probabilidades_finales()
        logger.info(f"Probabilidades cargadas: {len(df_prob)} partidos")
        
        # 2. Aplicar regla de draw propensity
        df_adjusted = aplicar_regla_draw_propensity(df_prob)
        
        # 3. Verificar validez de probabilidades
        df_verified = verificar_probabilidades_validas(df_adjusted)
        
        # 4. Proyectar a rangos globales si es necesario
        df_projected = proyectar_rangos_globales(df_verified)
        
        # 5. Marcar flags de draw propensity
        df_final = marcar_draw_propensity_flags(df_projected)
        
        # 6. Guardar resultados
        guardar_prob_draw(df_final)
        
        # 7. Estadísticas finales
        logger.info("=== ESTADÍSTICAS FINALES ===")
        
        prob_cols = ['p_final_L', 'p_final_E', 'p_final_V']
        for col in prob_cols:
            resultado = col[-1]
            media = df_final[col].mean()
            std = df_final[col].std()
            logger.info(f"Prob {resultado}: media={media:.3f}, std={std:.3f}")
        
        # Distribución global
        n_partidos = len(df_final)
        total_l = df_final['p_final_L'].sum()
        total_e = df_final['p_final_E'].sum()
        total_v = df_final['p_final_V'].sum()
        
        logger.info(f"Distribución global: L={total_l/n_partidos:.3f}, "
                   f"E={total_e/n_partidos:.3f}, V={total_v/n_partidos:.3f}")
        
        logger.info("=== PIPELINE DRAW PROPENSITY COMPLETADO ===")
        return df_final
        
    except Exception as e:
        logger.error(f"Error en pipeline draw propensity: {e}")
        
        # Fallback: usar probabilidades sin ajuste
        logger.warning("Usando probabilidades sin ajuste draw propensity...")
        
        try:
            df_prob = cargar_probabilidades_finales()
            # Agregar flag básico
            df_prob['draw_propensity_flag'] = df_prob['p_final_E'] > 0.33
            guardar_prob_draw(df_prob)
            return df_prob
        except:
            # Último fallback
            jornada = detectar_jornada()
            df_emergency = generar_probabilidades_default()
            df_emergency['draw_propensity_flag'] = False
            guardar_prob_draw(df_emergency)
            return df_emergency

if __name__ == "__main__":
    # Ejecutar pipeline
    resultado = pipeline_draw_propensity_completo()
    print(f"Pipeline draw propensity completado. Probabilidades ajustadas para {len(resultado)} partidos.")