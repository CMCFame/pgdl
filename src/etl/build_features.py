import pandas as pd
import numpy as np
import json
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger("build_features")

def verificar_y_reparar_estructura(df, tipo_archivo):
    """
    Verificar y reparar estructura de DataFrames para compatibilidad
    """
    logger.info(f"Verificando estructura de {tipo_archivo}")
    
    # Asegurar match_id
    if 'match_id' not in df.columns:
        if 'concurso_id' in df.columns and 'match_no' in df.columns:
            df['match_id'] = df['concurso_id'].astype(str) + '-' + df['match_no'].astype(str)
            logger.info(f"match_id generado para {tipo_archivo}")
        else:
            logger.error(f"No se puede generar match_id para {tipo_archivo}")
            raise ValueError(f"match_id requerido para {tipo_archivo}")
    
    # Validar que match_id no tenga duplicados
    if df['match_id'].duplicated().any():
        logger.warning(f"match_id duplicados en {tipo_archivo} - eliminando")
        df = df.drop_duplicates(subset=['match_id'], keep='first')
    
    return df

def normalizar_momios(df_odds):
    """Normalizar momios manteniendo compatibilidad total"""
    logger.info("Normalizando momios...")
    
    # Verificar estructura
    df_odds = verificar_y_reparar_estructura(df_odds, "odds")
    
    # Verificar columnas de odds
    required_odds = ['odds_L', 'odds_E', 'odds_V']
    missing_odds = [col for col in required_odds if col not in df_odds.columns]
    
    if missing_odds:
        logger.error(f"Columnas de odds faltantes: {missing_odds}")
        raise ValueError(f"Columnas faltantes: {missing_odds}")
    
    # Normalización
    inv_sum = 1 / df_odds[required_odds]
    row_sum = inv_sum.sum(axis=1)
    
    for col in ['L', 'E', 'V']:
        df_odds[f'p_raw_{col}'] = (1 / df_odds[f'odds_{col}']) / row_sum
    
    logger.info("Momios normalizados exitosamente")
    return df_odds

def merge_fuentes(df_progol, df_odds, previas_json, df_elo=None, df_squad=None):
    """Merge robusto que garantiza compatibilidad"""
    logger.info("Iniciando merge de fuentes...")
    
    # Verificar estructuras base
    df_progol = verificar_y_reparar_estructura(df_progol, "progol")
    df_odds = verificar_y_reparar_estructura(df_odds, "odds")
    
    # Merge principal por match_id
    df = df_progol.merge(df_odds, on='match_id', how='left', suffixes=('', '_odds'))
    logger.info(f"Merge Progol+Odds: {len(df)} registros")
    
    # Verificar que el merge fue exitoso
    missing_odds = df['p_raw_L'].isnull().sum()
    if missing_odds > 0:
        logger.warning(f"{missing_odds} partidos sin odds - rellenando con valores por defecto")
        df['p_raw_L'] = df['p_raw_L'].fillna(0.33)
        df['p_raw_E'] = df['p_raw_E'].fillna(0.33) 
        df['p_raw_V'] = df['p_raw_V'].fillna(0.34)
    
    # Merge Previas
    if previas_json:
        try:
            df_previas = pd.DataFrame(previas_json)
            if 'match_id' in df_previas.columns:
                df = df.merge(df_previas, on='match_id', how='left', suffixes=('', '_prev'))
                logger.info("Merge previas exitoso")
            else:
                logger.warning("Previas sin match_id - usando valores por defecto")
        except Exception as e:
            logger.warning(f"Error en merge previas: {e}")
    
    # Merge ELO con manejo robusto
    if df_elo is not None:
        try:
            if 'match_id' in df_elo.columns:
                # Merge por match_id
                elo_cols = ['match_id', 'elo_home', 'elo_away', 'factor_local']
                elo_cols = [col for col in elo_cols if col in df_elo.columns]
                df = df.merge(df_elo[elo_cols], on='match_id', how='left', suffixes=('', '_elo'))
            else:
                # Merge por equipos
                merge_cols = ['home', 'away']
                if 'fecha' in df.columns and 'fecha' in df_elo.columns:
                    merge_cols.append('fecha')
                
                elo_cols = ['elo_home', 'elo_away', 'factor_local'] + merge_cols
                elo_cols = [col for col in elo_cols if col in df_elo.columns]
                df = df.merge(df_elo[elo_cols], on=merge_cols, how='left', suffixes=('', '_elo'))
            
            logger.info("Merge ELO exitoso")
        except Exception as e:
            logger.warning(f"Error en merge ELO: {e}")
    
    # Merge Squad Values con manejo robusto
    if df_squad is not None:
        try:
            # Detectar nombre de columna de valor
            value_col = None
            for col in ['squad_value', 'market_value', 'value']:
                if col in df_squad.columns:
                    value_col = col
                    break
            
            if value_col:
                # Merge para equipos locales
                df = df.merge(
                    df_squad[['team', value_col, 'avg_age']].rename(columns={
                        value_col: 'squad_value_home',
                        'avg_age': 'avg_age_home'
                    }),
                    left_on='home', 
                    right_on='team', 
                    how='left'
                ).drop('team', axis=1)
                
                # Merge para equipos visitantes
                df = df.merge(
                    df_squad[['team', value_col, 'avg_age']].rename(columns={
                        value_col: 'squad_value_away', 
                        'avg_age': 'avg_age_away'
                    }),
                    left_on='away',
                    right_on='team',
                    how='left'
                ).drop('team', axis=1)
                
                logger.info("Merge Squad Values exitoso")
            else:
                logger.warning("Squad Values sin columna de valor reconocida")
        except Exception as e:
            logger.warning(f"Error en merge Squad Values: {e}")
    
    # Limpiar columnas duplicadas
    df = df.loc[:,~df.columns.duplicated()]
    
    logger.info(f"Merge completado: {len(df)} registros, {len(df.columns)} columnas")
    return df

def construir_features(df):
    """Construir features garantizando todas las columnas necesarias"""
    logger.info("Construyendo features...")
    
    # === FEATURES DE FORMA ===
    if all(col in df.columns for col in ['form_H', 'form_A']):
        df['form_H'] = df['form_H'].fillna('NNNNN')
        df['form_A'] = df['form_A'].fillna('NNNNN')
        
        df['gf5_H'] = df['form_H'].str.count('W') * 3 + df['form_H'].str.count('D')
        df['gf5_A'] = df['form_A'].str.count('W') * 3 + df['form_A'].str.count('D')
        df['delta_forma'] = df['gf5_H'] - df['gf5_A']
    else:
        df['form_H'] = 'NNNNN'
        df['form_A'] = 'NNNNN'
        df['gf5_H'] = 7.5
        df['gf5_A'] = 7.5
        df['delta_forma'] = 0.0

    # === H2H RATIO ===
    if all(col in df.columns for col in ['h2h_H', 'h2h_E', 'h2h_A']):
        df['h2h_H'] = df['h2h_H'].fillna(1)
        df['h2h_E'] = df['h2h_E'].fillna(1)
        df['h2h_A'] = df['h2h_A'].fillna(1)
        
        h_sum = df['h2h_H'] + df['h2h_E'] + df['h2h_A']
        df['h2h_ratio'] = (df['h2h_H'] - df['h2h_A']) / h_sum.replace(0, np.nan)
        df['h2h_ratio'] = df['h2h_ratio'].fillna(0)
    else:
        df['h2h_H'] = 1
        df['h2h_E'] = 1
        df['h2h_A'] = 1
        df['h2h_ratio'] = 0.0

    # === ELO DIFFERENTIAL ===
    if all(col in df.columns for col in ['elo_home', 'elo_away']):
        df['elo_home'] = df['elo_home'].fillna(1500)
        df['elo_away'] = df['elo_away'].fillna(1500)
        df['elo_diff'] = df['elo_home'] - df['elo_away']
    else:
        df['elo_home'] = 1500
        df['elo_away'] = 1500
        df['elo_diff'] = 0.0

    # === FACTOR LOCAL ===
    if 'factor_local' not in df.columns:
        df['factor_local'] = 0.45  # Valor por defecto

    # === LESIONES ===
    if all(col in df.columns for col in ['inj_H', 'inj_A']):
        df['inj_H'] = df['inj_H'].fillna(0)
        df['inj_A'] = df['inj_A'].fillna(0)
        df['inj_weight'] = (df['inj_H'] + df['inj_A']) / 11
    else:
        df['inj_H'] = 0
        df['inj_A'] = 0
        df['inj_weight'] = 0.0

    # === FLAGS CONTEXTUALES ===
    if 'context_flag' in df.columns:
        df['context_flag'] = df['context_flag'].fillna('[]')
        
        def extract_flags(x):
            if isinstance(x, list):
                return x
            elif isinstance(x, str):
                try:
                    return eval(x) if x.startswith('[') else [x] if x else []
                except:
                    return []
            else:
                return []
        
        df['context_flag'] = df['context_flag'].apply(extract_flags)
        df['is_final'] = df['context_flag'].apply(lambda x: any('final' in str(flag).lower() for flag in x))
        df['is_derby'] = df['context_flag'].apply(lambda x: any('derbi' in str(flag).lower() for flag in x))
    else:
        df['context_flag'] = [[] for _ in range(len(df))]
        df['is_final'] = False
        df['is_derby'] = False

    # === DRAW PROPENSITY FLAG ===
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
        df['squad_value_home'] = df['squad_value_home'].fillna(10.0)
        df['squad_value_away'] = df['squad_value_away'].fillna(10.0)
        df['value_diff'] = df['squad_value_home'] - df['squad_value_away']
    else:
        df['squad_value_home'] = 10.0
        df['squad_value_away'] = 10.0
        df['value_diff'] = 0.0

    # === VARIABLES ADICIONALES REQUERIDAS POR MÓDULOS POSTERIORES ===
    
    # Variables que esperan los módulos de modelado
    required_features = [
        'liga', 'home', 'away', 'fecha', 'concurso_id', 'match_no',
        'p_raw_L', 'p_raw_E', 'p_raw_V',
        'delta_forma', 'h2h_ratio', 'elo_diff', 'inj_weight',
        'is_final', 'is_derby', 'draw_propensity_flag', 'value_diff',
        'factor_local', 'elo_home', 'elo_away'
    ]
    
    # Asegurar que todas las features requeridas existan
    for feature in required_features:
        if feature not in df.columns:
            if feature in ['liga', 'home', 'away']:
                df[feature] = f'default_{feature}'
            elif feature == 'fecha':
                df[feature] = pd.to_datetime('2025-01-01')
            elif feature in ['concurso_id', 'match_no']:
                df[feature] = 1
            else:
                df[feature] = 0.0
            
            logger.warning(f"Feature faltante {feature} - agregada con valor por defecto")
    
    # === VALIDACIÓN FINAL ===
    if 'match_id' not in df.columns:
        logger.error("match_id no encontrado después de construir features")
        raise ValueError("match_id requerido")
    
    # Verificar que no hay valores infinitos o NaN en features críticas
    numeric_features = df.select_dtypes(include=[np.number]).columns
    for col in numeric_features:
        if df[col].isnull().any():
            df[col] = df[col].fillna(0)
            logger.warning(f"NaN values en {col} rellenados con 0")
        
        if np.isinf(df[col]).any():
            df[col] = df[col].replace([np.inf, -np.inf], 0)
            logger.warning(f"Valores infinitos en {col} reemplazados con 0")
    
    logger.info(f"Features construidos exitosamente: {len(df)} partidos, {len(df.columns)} features")
    
    return df

def guardar_features(df, output_path):
    """Guardar features con validación"""
    try:
        # Crear directorio si no existe
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Intentar guardar como feather primero
        df.to_feather(output_path)
        logger.info(f"Features guardados como feather: {output_path}")
        
        # También guardar como CSV para compatibilidad
        csv_path = str(output_path).replace('.feather', '.csv')
        df.to_csv(csv_path, index=False)
        logger.info(f"Features guardados como CSV: {csv_path}")
        
    except Exception as e:
        # Fallback: solo CSV
        csv_path = str(output_path).replace('.feather', '.csv')
        df.to_csv(csv_path, index=False)
        logger.warning(f"Feather falló, guardado solo como CSV: {csv_path}")

# === FUNCIÓN PRINCIPAL MEJORADA ===
def build_features_pipeline():
    """Pipeline principal con detección automática"""
    logger.info("Iniciando pipeline build_features mejorado")
    
    try:
        # 1. Detectar jornada automáticamente
        jornada = detectar_jornada()
        logger.info(f"Jornada detectada: {jornada}")
        
        # 2. Cargar archivos principales
        df_progol = cargar_progol_auto()
        df_odds = cargar_odds_auto()
        df_odds = normalizar_momios(df_odds)
        
        # 3. Cargar datos opcionales
        previas_json = cargar_previas_auto(jornada)
        df_elo = cargar_elo_auto(jornada)
        df_squad = cargar_squad_auto(jornada)
        
        # 4. Merge y construcción de features
        df_merged = merge_fuentes(df_progol, df_odds, previas_json, df_elo, df_squad)
        df_final = construir_features(df_merged)
        
        # 5. Guardar resultados
        output_path = f"data/processed/match_features_{jornada}.feather"
        guardar_features(df_final, output_path)
        
        logger.info("Pipeline build_features completado exitosamente")
        return df_final
        
    except Exception as e:
        logger.error(f"Error en pipeline build_features: {e}")
        raise

def detectar_jornada():
    """Detectar jornada desde archivos disponibles"""
    raw_path = Path("data/raw")
    
    # Buscar en archivos CSV
    for archivo in raw_path.glob("*.csv"):
        try:
            if 'progol' in archivo.name.lower():
                df_temp = pd.read_csv(archivo, nrows=1)
                if 'concurso_id' in df_temp.columns:
                    return int(df_temp['concurso_id'].iloc[0])
        except:
            continue
    
    return 2287  # Fallback

def cargar_progol_auto():
    """Cargar Progol automáticamente"""
    from src.etl.ingest_csv import cargar_progol
    return cargar_progol()

def cargar_odds_auto():
    """Cargar odds automáticamente"""
    from src.etl.scrape_odds import cargar_odds
    return cargar_odds()

def cargar_previas_auto(jornada):
    """Cargar previas automáticamente"""
    json_path = Path("data/json_previas")
    
    posibles = [
        json_path / f"previas_{jornada}.json",
        json_path / "previas.json"
    ]
    
    for archivo in posibles:
        if archivo.exists():
            try:
                with open(archivo, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                continue
    
    return []

def cargar_elo_auto(jornada):
    """Cargar ELO automáticamente"""
    raw_path = Path("data/raw")
    
    posibles = [
        raw_path / f"elo_{jornada}.csv",
        raw_path / "elo.csv"
    ]
    
    for archivo in posibles:
        if archivo.exists():
            try:
                return pd.read_csv(archivo)
            except:
                continue
    
    return None

def cargar_squad_auto(jornada):
    """Cargar squad values automáticamente"""
    raw_path = Path("data/raw")
    
    posibles = [
        raw_path / f"squad_value_{jornada}.csv",
        raw_path / "squad_value.csv",
        raw_path / f"squad_values_{jornada}.csv"
    ]
    
    for archivo in posibles:
        if archivo.exists():
            try:
                return pd.read_csv(archivo)
            except:
                continue
    
    return None

if __name__ == "__main__":
    build_features_pipeline()