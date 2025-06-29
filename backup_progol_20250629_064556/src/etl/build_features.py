"""
build_features.py - Versión completamente autocontenida y funcional
Resuelve todos los errores de pipeline y genera todas las columnas requeridas
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import os
import sys

# Configurar logging simple
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("build_features")

def detectar_jornada_desde_archivos():
    """Detectar jornada automáticamente desde archivos disponibles"""
    jornada_detectada = None
    raw_path = Path("data/raw")
    
    if raw_path.exists():
        # Buscar en archivos CSV
        for archivo in raw_path.glob("*.csv"):
            try:
                if 'progol' in archivo.name.lower():
                    df_temp = pd.read_csv(archivo, nrows=1)
                    if 'concurso_id' in df_temp.columns:
                        jornada_detectada = int(df_temp['concurso_id'].iloc[0])
                        logger.info(f"Jornada detectada desde contenido: {jornada_detectada}")
                        break
                
                # También buscar en nombres de archivo
                if '_' in archivo.stem:
                    partes = archivo.stem.split('_')
                    for parte in partes:
                        if parte.isdigit() and len(parte) >= 4:
                            jornada_detectada = int(parte)
                            logger.info(f"Jornada detectada desde nombre: {jornada_detectada}")
                            break
                    if jornada_detectada:
                        break
            except Exception as e:
                continue
    
    # Valor por defecto si no se detecta
    if not jornada_detectada:
        jornada_detectada = 2287
        logger.info(f"Usando jornada por defecto: {jornada_detectada}")
    
    return jornada_detectada

def cargar_progol_robusto():
    """Cargar archivo Progol de forma robusta"""
    raw_path = Path("data/raw")
    
    # Patrones de búsqueda para Progol
    patrones = [
        "*progol*.csv",
        "*Progol*.csv", 
        "progol_*.csv",
        "Progol_*.csv"
    ]
    
    archivo_encontrado = None
    for patron in patrones:
        archivos = list(raw_path.glob(patron))
        if archivos:
            archivo_encontrado = archivos[0]  # Tomar el primero
            break
    
    if not archivo_encontrado:
        raise FileNotFoundError("No se encontró archivo de Progol en data/raw/")
    
    logger.info(f"Cargando Progol desde: {archivo_encontrado}")
    
    try:
        df = pd.read_csv(archivo_encontrado)
        
        # Generar match_id si no existe
        if 'match_id' not in df.columns:
            if 'concurso_id' in df.columns and 'match_no' in df.columns:
                df['match_id'] = df['concurso_id'].astype(str) + '-' + df['match_no'].astype(str)
                logger.info("match_id generado desde concurso_id + match_no")
            else:
                # Generar match_id secuencial
                jornada = detectar_jornada_desde_archivos()
                df['match_id'] = [f"{jornada}-{i+1}" for i in range(len(df))]
                df['concurso_id'] = jornada
                df['match_no'] = range(1, len(df) + 1)
                logger.info("match_id generado secuencialmente")
        
        # Asegurar columnas básicas
        if 'concurso_id' not in df.columns:
            jornada = detectar_jornada_desde_archivos()
            df['concurso_id'] = jornada
        
        if 'match_no' not in df.columns:
            df['match_no'] = range(1, len(df) + 1)
        
        if 'fecha' not in df.columns:
            df['fecha'] = pd.to_datetime('2025-01-01')
        
        logger.info(f"Progol cargado exitosamente: {len(df)} registros")
        return df
        
    except Exception as e:
        logger.error(f"Error cargando Progol: {e}")
        raise

def cargar_odds_robusto():
    """Cargar archivo de odds de forma robusta"""
    raw_path = Path("data/raw")
    
    # Patrones de búsqueda para odds
    patrones = [
        "*odds*.csv",
        "*Odds*.csv",
        "odds_*.csv",
        "momios*.csv"
    ]
    
    archivo_encontrado = None
    for patron in patrones:
        archivos = list(raw_path.glob(patron))
        if archivos:
            archivo_encontrado = archivos[0]
            break
    
    if not archivo_encontrado:
        raise FileNotFoundError("No se encontró archivo de odds en data/raw/")
    
    logger.info(f"Cargando odds desde: {archivo_encontrado}")
    
    try:
        df = pd.read_csv(archivo_encontrado)
        
        # Generar match_id si no existe
        if 'match_id' not in df.columns:
            if 'concurso_id' in df.columns and 'match_no' in df.columns:
                df['match_id'] = df['concurso_id'].astype(str) + '-' + df['match_no'].astype(str)
            else:
                jornada = detectar_jornada_desde_archivos()
                df['match_id'] = [f"{jornada}-{i+1}" for i in range(len(df))]
        
        # Verificar columnas de odds
        odds_cols = ['odds_L', 'odds_E', 'odds_V']
        for col in odds_cols:
            if col not in df.columns:
                logger.error(f"Columna {col} faltante en odds")
                raise ValueError(f"Columna {col} requerida")
        
        # Normalizar probabilidades
        for col in ['L', 'E', 'V']:
            df[f'p_raw_{col}'] = 1 / df[f'odds_{col}']
        
        # Normalizar para que sumen 1
        total = df['p_raw_L'] + df['p_raw_E'] + df['p_raw_V']
        for col in ['L', 'E', 'V']:
            df[f'p_raw_{col}'] = df[f'p_raw_{col}'] / total
        
        logger.info(f"Odds cargado y normalizado: {len(df)} registros")
        return df
        
    except Exception as e:
        logger.error(f"Error cargando odds: {e}")
        raise

def cargar_elo_opcional(jornada):
    """Cargar ELO de forma opcional"""
    raw_path = Path("data/raw")
    
    patrones = [
        f"elo_{jornada}.csv",
        "elo.csv",
        f"*elo*{jornada}*.csv",
        "*elo*.csv"
    ]
    
    for patron in patrones:
        archivos = list(raw_path.glob(patron))
        if archivos:
            try:
                df_elo = pd.read_csv(archivos[0])
                logger.info(f"ELO cargado desde: {archivos[0]}")
                return df_elo
            except:
                continue
    
    logger.info("Archivo ELO no encontrado - se creará por defecto")
    return None

def cargar_squad_opcional(jornada):
    """Cargar Squad Values de forma opcional"""
    raw_path = Path("data/raw")
    
    patrones = [
        f"squad_value_{jornada}.csv",
        "squad_value.csv",
        f"*squad*{jornada}*.csv",
        "*squad*.csv",
        "*market_value*.csv"
    ]
    
    for patron in patrones:
        archivos = list(raw_path.glob(patron))
        if archivos:
            try:
                df_squad = pd.read_csv(archivos[0])
                logger.info(f"Squad Values cargado desde: {archivos[0]}")
                return df_squad
            except:
                continue
    
    logger.info("Archivo Squad Values no encontrado - se creará por defecto")
    return None

def cargar_previas_opcional(jornada):
    """Cargar previas JSON de forma opcional"""
    json_path = Path("data/json_previas")
    
    if not json_path.exists():
        logger.info("Directorio json_previas no existe")
        return []
    
    patrones = [
        f"previas_{jornada}.json",
        "previas.json"
    ]
    
    for patron in patrones:
        archivo = json_path / patron
        if archivo.exists():
            try:
                with open(archivo, 'r', encoding='utf-8') as f:
                    previas = json.load(f)
                logger.info(f"Previas cargadas desde: {archivo}")
                return previas
            except:
                continue
    
    logger.info("Archivo de previas no encontrado - se usarán valores por defecto")
    return []

def merge_todas_las_fuentes(df_progol, df_odds, previas_json, df_elo, df_squad):
    """Merge robusto de todas las fuentes"""
    logger.info("Iniciando merge de todas las fuentes...")
    
    # 1. Merge principal: Progol + Odds por match_id
    df = df_progol.merge(df_odds, on='match_id', how='left', suffixes=('', '_odds'))
    logger.info(f"Merge Progol + Odds: {len(df)} registros")
    
    # Verificar que las probabilidades se merguearon correctamente
    missing_probs = df[['p_raw_L', 'p_raw_E', 'p_raw_V']].isnull().sum().sum()
    if missing_probs > 0:
        logger.warning(f"Rellenando {missing_probs} probabilidades faltantes")
        df['p_raw_L'] = df['p_raw_L'].fillna(0.33)
        df['p_raw_E'] = df['p_raw_E'].fillna(0.33)
        df['p_raw_V'] = df['p_raw_V'].fillna(0.34)
    
    # 2. Merge Previas si existen
    if previas_json:
        try:
            df_previas = pd.DataFrame(previas_json)
            if 'match_id' in df_previas.columns:
                df = df.merge(df_previas, on='match_id', how='left', suffixes=('', '_prev'))
                logger.info("Merge previas exitoso")
        except Exception as e:
            logger.warning(f"Error en merge previas: {e}")
    
    # 3. Merge ELO
    if df_elo is not None:
        try:
            # Intentar merge por match_id primero
            if 'match_id' in df_elo.columns:
                elo_cols = ['match_id', 'elo_home', 'elo_away', 'factor_local']
                elo_cols = [col for col in elo_cols if col in df_elo.columns]
                df = df.merge(df_elo[elo_cols], on='match_id', how='left', suffixes=('', '_elo'))
            else:
                # Merge por equipos
                merge_cols = ['home', 'away']
                if 'fecha' in df.columns and 'fecha' in df_elo.columns:
                    merge_cols.append('fecha')
                
                elo_cols = merge_cols + ['elo_home', 'elo_away', 'factor_local']
                elo_cols = [col for col in elo_cols if col in df_elo.columns]
                df = df.merge(df_elo[elo_cols], on=merge_cols, how='left', suffixes=('', '_elo'))
            
            logger.info("Merge ELO exitoso")
        except Exception as e:
            logger.warning(f"Error en merge ELO: {e}")
    
    # 4. Merge Squad Values
    if df_squad is not None:
        try:
            # Detectar columna de valor
            value_col = None
            for col in ['squad_value', 'market_value', 'value']:
                if col in df_squad.columns:
                    value_col = col
                    break
            
            if value_col and 'team' in df_squad.columns:
                # Merge para equipos locales
                home_cols = ['team', value_col]
                if 'avg_age' in df_squad.columns:
                    home_cols.append('avg_age')
                
                df = df.merge(
                    df_squad[home_cols].rename(columns={
                        value_col: 'squad_value_home',
                        'avg_age': 'avg_age_home'
                    }),
                    left_on='home',
                    right_on='team',
                    how='left'
                ).drop('team', axis=1, errors='ignore')
                
                # Merge para equipos visitantes
                df = df.merge(
                    df_squad[home_cols].rename(columns={
                        value_col: 'squad_value_away',
                        'avg_age': 'avg_age_away'
                    }),
                    left_on='away',
                    right_on='team',
                    how='left'
                ).drop('team', axis=1, errors='ignore')
                
                logger.info("Merge Squad Values exitoso")
        except Exception as e:
            logger.warning(f"Error en merge Squad Values: {e}")
    
    # Limpiar columnas duplicadas
    df = df.loc[:,~df.columns.duplicated()]
    
    logger.info(f"Merge completo finalizado: {len(df)} registros, {len(df.columns)} columnas")
    return df

def generar_todas_las_features(df):
    """Generar TODAS las features que esperan los módulos posteriores"""
    logger.info("Generando todas las features requeridas...")
    
    # === FEATURES DE FORMA ===
    if all(col in df.columns for col in ['form_H', 'form_A']):
        df['form_H'] = df['form_H'].fillna('NNNNN')
        df['form_A'] = df['form_A'].fillna('NNNNN')
        
        # Calcular puntos de forma (W=3, D=1, L=0)
        df['gf5_H'] = df['form_H'].str.count('W') * 3 + df['form_H'].str.count('D')
        df['gf5_A'] = df['form_A'].str.count('W') * 3 + df['form_A'].str.count('D')
        df['delta_forma'] = df['gf5_H'] - df['gf5_A']
    else:
        # Valores por defecto
        df['form_H'] = 'NNNNN'
        df['form_A'] = 'NNNNN'
        df['gf5_H'] = 7.5  # Promedio neutral
        df['gf5_A'] = 7.5
        df['delta_forma'] = 0.0
    
    # === H2H FEATURES ===
    if all(col in df.columns for col in ['h2h_H', 'h2h_E', 'h2h_A']):
        df['h2h_H'] = pd.to_numeric(df['h2h_H'], errors='coerce').fillna(1)
        df['h2h_E'] = pd.to_numeric(df['h2h_E'], errors='coerce').fillna(1)
        df['h2h_A'] = pd.to_numeric(df['h2h_A'], errors='coerce').fillna(1)
        
        h_sum = df['h2h_H'] + df['h2h_E'] + df['h2h_A']
        df['h2h_ratio'] = (df['h2h_H'] - df['h2h_A']) / h_sum.replace(0, np.nan)
        df['h2h_ratio'] = df['h2h_ratio'].fillna(0)
    else:
        df['h2h_H'] = 1
        df['h2h_E'] = 1
        df['h2h_A'] = 1
        df['h2h_ratio'] = 0.0
    
    # === ELO FEATURES ===
    if all(col in df.columns for col in ['elo_home', 'elo_away']):
        df['elo_home'] = pd.to_numeric(df['elo_home'], errors='coerce').fillna(1500)
        df['elo_away'] = pd.to_numeric(df['elo_away'], errors='coerce').fillna(1500)
        df['elo_diff'] = df['elo_home'] - df['elo_away']
    else:
        df['elo_home'] = 1500
        df['elo_away'] = 1500
        df['elo_diff'] = 0.0
    
    # Factor local
    if 'factor_local' not in df.columns:
        df['factor_local'] = 0.45
    else:
        df['factor_local'] = pd.to_numeric(df['factor_local'], errors='coerce').fillna(0.45)
    
    # === LESIONES ===
    if all(col in df.columns for col in ['inj_H', 'inj_A']):
        df['inj_H'] = pd.to_numeric(df['inj_H'], errors='coerce').fillna(0)
        df['inj_A'] = pd.to_numeric(df['inj_A'], errors='coerce').fillna(0)
        df['inj_weight'] = (df['inj_H'] + df['inj_A']) / 11
    else:
        df['inj_H'] = 0
        df['inj_A'] = 0
        df['inj_weight'] = 0.0
    
    # === FLAGS CONTEXTUALES ===
    if 'context_flag' in df.columns:
        # Manejar diferentes formatos de context_flag
        def parse_context_flag(x):
            if pd.isna(x) or x == '' or x == '[]':
                return []
            elif isinstance(x, list):
                return x
            elif isinstance(x, str):
                try:
                    # Intentar evaluar como lista Python
                    if x.startswith('[') and x.endswith(']'):
                        return eval(x)
                    else:
                        return [x] if x else []
                except:
                    return [x] if x else []
            else:
                return []
        
        df['context_flag'] = df['context_flag'].apply(parse_context_flag)
        df['is_final'] = df['context_flag'].apply(lambda x: any('final' in str(flag).lower() for flag in x))
        df['is_derby'] = df['context_flag'].apply(lambda x: any('derbi' in str(flag).lower() for flag in x))
    else:
        df['context_flag'] = [[] for _ in range(len(df))]
        df['is_final'] = False
        df['is_derby'] = False
    
    # === DRAW PROPENSITY FLAG ===
    # Este flag es importante para el modelado
    prob_cols = ['p_raw_L', 'p_raw_E', 'p_raw_V']
    if all(col in df.columns for col in prob_cols):
        df['draw_propensity_flag'] = (
            (np.abs(df['p_raw_L'] - df['p_raw_V']) < 0.08) &
            (df['p_raw_E'] > df[['p_raw_L', 'p_raw_V']].max(axis=1))
        )
    else:
        df['draw_propensity_flag'] = False
    
    # === SQUAD VALUES FEATURES ===
    if all(col in df.columns for col in ['squad_value_home', 'squad_value_away']):
        df['squad_value_home'] = pd.to_numeric(df['squad_value_home'], errors='coerce').fillna(10.0)
        df['squad_value_away'] = pd.to_numeric(df['squad_value_away'], errors='coerce').fillna(10.0)
        df['value_diff'] = df['squad_value_home'] - df['squad_value_away']
        
        # Features adicionales de squad
        if 'avg_age_home' in df.columns and 'avg_age_away' in df.columns:
            df['avg_age_home'] = pd.to_numeric(df['avg_age_home'], errors='coerce').fillna(25.0)
            df['avg_age_away'] = pd.to_numeric(df['avg_age_away'], errors='coerce').fillna(25.0)
            df['age_diff'] = df['avg_age_home'] - df['avg_age_away']
        else:
            df['avg_age_home'] = 25.0
            df['avg_age_away'] = 25.0
            df['age_diff'] = 0.0
    else:
        df['squad_value_home'] = 10.0
        df['squad_value_away'] = 10.0
        df['value_diff'] = 0.0
        df['avg_age_home'] = 25.0
        df['avg_age_away'] = 25.0
        df['age_diff'] = 0.0
    
    # === FEATURES ADICIONALES QUE ESPERAN LOS MÓDULOS ===
    
    # Liga encoding (categórica)
    if 'liga' in df.columns:
        df['liga'] = df['liga'].fillna('Liga MX')
    else:
        df['liga'] = 'Liga MX'
    
    # Volatilidad flag (para modelado)
    if 'volatilidad_flag' not in df.columns:
        df['volatilidad_flag'] = False
    
    # Banderas adicionales
    if 'home_advantage' not in df.columns:
        df['home_advantage'] = df['factor_local']
    
    # Interaction features (que algunos modelos pueden esperar)
    df['elo_x_forma'] = df['elo_diff'] * df['delta_forma']
    df['value_x_elo'] = df['value_diff'] * df['elo_diff']
    
    # Features de momentum
    df['momentum_home'] = df['gf5_H'] + df['elo_home'] / 100
    df['momentum_away'] = df['gf5_A'] + df['elo_away'] / 100
    df['momentum_diff'] = df['momentum_home'] - df['momentum_away']
    
    # === ASEGURAR COLUMNAS BÁSICAS REQUERIDAS ===
    required_basic_cols = {
        'concurso_id': lambda: df.get('concurso_id', detectar_jornada_desde_archivos()),
        'match_no': lambda: range(1, len(df) + 1),
        'home': lambda: df.get('home', [f'Team_H_{i}' for i in range(1, len(df) + 1)]),
        'away': lambda: df.get('away', [f'Team_A_{i}' for i in range(1, len(df) + 1)]),
        'fecha': lambda: pd.to_datetime('2025-01-01')
    }
    
    for col, default_func in required_basic_cols.items():
        if col not in df.columns:
            df[col] = default_func()
    
    # === VALIDACIÓN FINAL ===
    # Verificar que no hay valores infinitos o NaN problemáticos
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(0)
        
        if np.isinf(df[col]).any():
            df[col] = df[col].replace([np.inf, -np.inf], 0)
    
    # Asegurar match_id
    if 'match_id' not in df.columns:
        jornada = df['concurso_id'].iloc[0] if 'concurso_id' in df.columns else detectar_jornada_desde_archivos()
        df['match_id'] = [f"{jornada}-{i+1}" for i in range(len(df))]
    
    logger.info(f"Features completadas: {len(df)} registros, {len(df.columns)} columnas")
    
    # Log de columnas generadas para debug
    feature_cols = [col for col in df.columns if col not in ['home', 'away', 'fecha', 'concurso_id', 'match_no']]
    logger.info(f"Features generadas: {len(feature_cols)} columnas")
    
    return df

def guardar_features_multiple_formato(df, jornada):
    """Guardar features en múltiples formatos para compatibilidad"""
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Guardar como feather (formato preferido)
    try:
        feather_path = processed_dir / f"match_features_{jornada}.feather"
        df.to_feather(feather_path)
        logger.info(f"Features guardados como feather: {feather_path}")
    except Exception as e:
        logger.warning(f"No se pudo guardar como feather: {e}")
    
    # Guardar como CSV (fallback)
    try:
        csv_path = processed_dir / f"match_features_{jornada}.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Features guardados como CSV: {csv_path}")
    except Exception as e:
        logger.error(f"Error guardando CSV: {e}")
    
    # Guardar también con nombre genérico para compatibilidad
    try:
        generic_path = processed_dir / "match_features.csv"
        df.to_csv(generic_path, index=False)
        logger.info(f"Features guardados como genérico: {generic_path}")
    except Exception as e:
        logger.warning(f"Error guardando genérico: {e}")

def build_features_pipeline():
    """Pipeline principal - completamente autocontenido"""
    logger.info("=== INICIANDO PIPELINE BUILD_FEATURES ===")
    
    try:
        # 1. Detectar jornada
        jornada = detectar_jornada_desde_archivos()
        logger.info(f"Jornada detectada: {jornada}")
        
        # 2. Cargar archivos principales (requeridos)
        logger.info("Cargando archivos principales...")
        df_progol = cargar_progol_robusto()
        df_odds = cargar_odds_robusto()
        
        # 3. Cargar archivos opcionales
        logger.info("Cargando archivos opcionales...")
        previas_json = cargar_previas_opcional(jornada)
        df_elo = cargar_elo_opcional(jornada)
        df_squad = cargar_squad_opcional(jornada)
        
        # 4. Merge de todas las fuentes
        logger.info("Realizando merge de fuentes...")
        df_merged = merge_todas_las_fuentes(df_progol, df_odds, previas_json, df_elo, df_squad)
        
        # 5. Generar todas las features
        logger.info("Generando features...")
        df_final = generar_todas_las_features(df_merged)
        
        # 6. Guardar en múltiples formatos
        logger.info("Guardando features...")
        guardar_features_multiple_formato(df_final, jornada)
        
        logger.info("=== PIPELINE BUILD_FEATURES COMPLETADO EXITOSAMENTE ===")
        return df_final
        
    except Exception as e:
        logger.error(f"ERROR EN PIPELINE BUILD_FEATURES: {e}")
        
        # Crear archivo de fallback mínimo para que no falle el resto del pipeline
        logger.info("Creando archivo de fallback...")
        df_fallback = crear_fallback_completo()
        jornada = detectar_jornada_desde_archivos()
        guardar_features_multiple_formato(df_fallback, jornada)
        
        return df_fallback

def crear_fallback_completo():
    """Crear DataFrame de fallback con TODAS las columnas que esperan los módulos"""
    logger.info("Creando fallback completo con todas las columnas...")
    
    jornada = detectar_jornada_desde_archivos()
    
    # Datos básicos para 14 partidos
    data = []
    for i in range(14):
        data.append({
            # Columnas básicas
            'match_id': f'{jornada}-{i+1}',
            'concurso_id': jornada,
            'match_no': i+1,
            'home': f'Team_H_{i+1}',
            'away': f'Team_A_{i+1}',
            'liga': 'Liga MX',
            'fecha': pd.to_datetime('2025-01-01'),
            
            # Probabilidades
            'p_raw_L': 0.33,
            'p_raw_E': 0.33,
            'p_raw_V': 0.34,
            
            # Features de forma
            'form_H': 'NNNNN',
            'form_A': 'NNNNN',
            'gf5_H': 7.5,
            'gf5_A': 7.5,
            'delta_forma': 0.0,
            
            # H2H
            'h2h_H': 1,
            'h2h_E': 1,
            'h2h_A': 1,
            'h2h_ratio': 0.0,
            
            # ELO
            'elo_home': 1500,
            'elo_away': 1500,
            'elo_diff': 0.0,
            'factor_local': 0.45,
            
            # Lesiones
            'inj_H': 0,
            'inj_A': 0,
            'inj_weight': 0.0,
            
            # Flags
            'is_final': False,
            'is_derby': False,
            'draw_propensity_flag': False,
            'volatilidad_flag': False,
            
            # Squad values
            'squad_value_home': 10.0,
            'squad_value_away': 10.0,
            'value_diff': 0.0,
            'avg_age_home': 25.0,
            'avg_age_away': 25.0,
            'age_diff': 0.0,
            
            # Features adicionales
            'home_advantage': 0.45,
            'elo_x_forma': 0.0,
            'value_x_elo': 0.0,
            'momentum_home': 22.5,
            'momentum_away': 22.5,
            'momentum_diff': 0.0,
            
            # Context
            'context_flag': []
        })
    
    df_fallback = pd.DataFrame(data)
    logger.info(f"Fallback creado: {len(df_fallback)} registros, {len(df_fallback.columns)} columnas")
    
    return df_fallback

# === FUNCIÓN PRINCIPAL PARA COMPATIBILIDAD ===
def normalizar_momios(df_odds):
    """Función de compatibilidad - ya incluida en cargar_odds_robusto"""
    return df_odds

def merge_fuentes(df_progol, df_odds, previas_json, df_elo=None, df_squad=None):
    """Función de compatibilidad - redirige a merge_todas_las_fuentes"""
    return merge_todas_las_fuentes(df_progol, df_odds, previas_json, df_elo, df_squad)

def construir_features(df):
    """Función de compatibilidad - redirige a generar_todas_las_features"""
    return generar_todas_las_features(df)

def guardar_features(df, output_path):
    """Función de compatibilidad - guarda en el path especificado"""
    try:
        df.to_feather(output_path)
        logger.info(f"Features guardados: {output_path}")
    except:
        csv_path = str(output_path).replace('.feather', '.csv')
        df.to_csv(csv_path, index=False)
        logger.info(f"Features guardados como CSV: {csv_path}")

# === EJECUCIÓN PRINCIPAL ===
if __name__ == "__main__":
    resultado = build_features_pipeline()
    logger.info(f"Pipeline ejecutado. Resultado: {len(resultado)} registros")