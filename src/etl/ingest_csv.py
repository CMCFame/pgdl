import pandas as pd
from pathlib import Path
from src.utils.logger import get_logger
#from src.etl.build_features import detectar_estructura_archivo, generar_match_id_dinamico, detectar_jornada_automaticamente

logger = get_logger("ingest_csv")

def detectar_jornada_simple():
    from pathlib import Path
    raw_path = Path("data/raw")
    for archivo in raw_path.glob("*.csv"):
        try:
            if 'progol' in archivo.name.lower():
                df_temp = pd.read_csv(archivo, nrows=1)
                if 'concurso_id' in df_temp.columns:
                    return int(df_temp['concurso_id'].iloc[0])
        except:
            continue
    return 2287

def detectar_archivo_progol():
    """
    Detecta automáticamente el archivo de Progol en data/raw/
    """
    raw_path = Path("data/raw")
    if not raw_path.exists():
        raise FileNotFoundError("Directorio data/raw no encontrado")
    
    # Patrones de búsqueda para archivo Progol
    patrones = [
        "progol*.csv",
        "Progol*.csv", 
        "PROGOL*.csv",
        "partidos*.csv",
        "matches*.csv"
    ]
    
    archivos_encontrados = []
    for patron in patrones:
        archivos_encontrados.extend(list(raw_path.glob(patron)))
    
    if not archivos_encontrados:
        # Buscar cualquier CSV que pueda ser Progol
        todos_csv = list(raw_path.glob("*.csv"))
        for archivo in todos_csv:
            try:
                df_test = pd.read_csv(archivo, nrows=5)
                info = detectar_estructura_archivo(df_test, archivo.name)
                if info['tipo'] in ['progol_prediccion', 'progol_historico']:
                    archivos_encontrados.append(archivo)
            except:
                continue
    
    if not archivos_encontrados:
        raise FileNotFoundError("No se encontró archivo de Progol válido en data/raw/")
    
    # Si hay múltiples, tomar el más reciente
    archivo_progol = max(archivos_encontrados, key=lambda x: x.stat().st_mtime)
    logger.info(f"Archivo Progol detectado: {archivo_progol.name}")
    
    return archivo_progol

def validar_progol_dinamico(df, info_estructura):
    """
    Valida archivo Progol según su estructura detectada
    """
    logger.info(f"Validando Progol tipo: {info_estructura['tipo']}")
    
    # Validaciones básicas
    if len(df) == 0:
        raise ValueError("Archivo Progol vacío")
    
    # Validaciones por tipo
    if info_estructura['tipo'] == 'progol_prediccion':
        # Para predicción: NO debe tener resultados
        cols_resultados = ['resultado', 'l_g', 'a_g']
        cols_encontradas = [col for col in cols_resultados if col in df.columns]
        
        if cols_encontradas:
            logger.warning(f"Archivo para predicción contiene columnas de resultados: {cols_encontradas}")
            logger.warning("Estas columnas serán ignoradas para predicción")
        
        # Debe tener columnas básicas
        cols_requeridas = ['home', 'away']
        cols_faltantes = [col for col in cols_requeridas if col not in df.columns]
        if cols_faltantes:
            raise ValueError(f"Columnas requeridas faltantes: {cols_faltantes}")
        
        logger.info("✅ Archivo válido para PREDICCIÓN")
    
    elif info_estructura['tipo'] == 'progol_historico':
        # Para histórico: debe tener resultados
        cols_requeridas = ['home', 'away', 'resultado']
        cols_faltantes = [col for col in cols_requeridas if col not in df.columns]
        if cols_faltantes:
            raise ValueError(f"Archivo histórico - columnas faltantes: {cols_faltantes}")
        
        # Validar consistencia de resultados si tiene goles
        if all(col in df.columns for col in ['l_g', 'a_g', 'resultado']):
            logger.info("Validando consistencia de resultados...")
            
            df_temp = df.copy()
            df_temp['resultado_calc'] = df_temp.apply(
                lambda r: "L" if r["l_g"] > r["a_g"] else "E" if r["l_g"] == r["a_g"] else "V", 
                axis=1
            )
            
            inconsistentes = (df_temp["resultado"] != df_temp["resultado_calc"]).sum()
            if inconsistentes > 0:
                logger.warning(f"{inconsistentes} resultados inconsistentes detectados y corregidos")
                df['resultado'] = df_temp['resultado_calc']
        
        logger.info("✅ Archivo válido para ANÁLISIS HISTÓRICO")
    
    else:
        logger.warning(f"Tipo de Progol no reconocido: {info_estructura['tipo']}")
    
    # Validar match_id
    if 'match_id' in df.columns:
        duplicados = df['match_id'].duplicated().sum()
        if duplicados > 0:
            logger.warning(f"{duplicados} match_id duplicados - eliminando duplicados")
            df = df.drop_duplicates(subset=['match_id'], keep='first')
    
    return df

def cargar_progol_dinamico(archivo_path=None):
    """
    Carga archivo Progol de forma completamente dinámica
    """
    if archivo_path is None:
        archivo_path = detectar_archivo_progol()
    
    logger.info(f"Cargando archivo Progol desde {archivo_path}")
    
    try:
        # Intentar detectar automáticamente el formato de fecha
        df = pd.read_csv(archivo_path)
        
        if 'fecha' in df.columns:
            # Intentar múltiples formatos de fecha
            formatos_fecha = [
                '%Y-%m-%d',
                '%d/%m/%Y', 
                '%m/%d/%Y',
                '%Y/%m/%d',
                '%d-%m-%Y',
                '%m-%d-%Y'
            ]
            
            fecha_parseada = False
            for formato in formatos_fecha:
                try:
                    df['fecha'] = pd.to_datetime(df['fecha'], format=formato)
                    fecha_parseada = True
                    logger.info(f"Fecha parseada con formato: {formato}")
                    break
                except:
                    continue
            
            if not fecha_parseada:
                try:
                    df['fecha'] = pd.to_datetime(df['fecha'])
                    logger.info("Fecha parseada automáticamente")
                except:
                    logger.warning("No se pudo parsear fecha - manteniendo como texto")
    
    except Exception as e:
        logger.error(f"Error leyendo archivo: {e}")
        raise
    
    # Detectar estructura
    info = detectar_estructura_archivo(df, archivo_path.name)
    logger.info(f"Estructura detectada: {info}")
    
    # Validar según estructura
    df = validar_progol_dinamico(df, info)
    
    # Generar match_id si es necesario
    jornada = detectar_jornada_automaticamente()
    df = generar_match_id_dinamico(df, jornada)
    
    logger.info(f"Archivo Progol cargado exitosamente: {len(df)} registros")
    return df, info

def guardar_progol_procesado(df, info_estructura):
    """
    Guarda el archivo Progol procesado
    """
    processed_path = Path("data/processed")
    processed_path.mkdir(parents=True, exist_ok=True)
    
    # Detectar jornada para el nombre
    jornada = detectar_jornada_automaticamente()
    
    # Nombre según tipo
    if info_estructura['tipo'] == 'progol_prediccion':
        filename = f"progol_clean_prediccion_{jornada}.csv"
    elif info_estructura['tipo'] == 'progol_historico':
        filename = f"progol_clean_historico_{jornada}.csv"
    else:
        filename = f"progol_clean_{jornada}.csv"
    
    output_path = processed_path / filename
    df.to_csv(output_path, index=False)
    
    logger.info(f"Archivo Progol procesado guardado: {output_path}")
    return output_path

def generar_reporte_progol(df, info_estructura):
    """
    Genera reporte de análisis del archivo Progol
    """
    reporte = {
        'tipo_archivo': info_estructura['tipo'],
        'total_registros': len(df),
        'total_columnas': len(df.columns),
        'columnas': list(df.columns),
        'jornada_detectada': detectar_jornada_automaticamente(),
        'fecha_rango': None,
        'equipos_unicos': None,
        'ligas_detectadas': None,
        'problemas': info_estructura.get('problemas', [])
    }
    
    # Análisis de fechas
    if 'fecha' in df.columns:
        try:
            reporte['fecha_rango'] = {
                'desde': str(df['fecha'].min()),
                'hasta': str(df['fecha'].max()),
                'fechas_unicas': df['fecha'].nunique()
            }
        except:
            reporte['fecha_rango'] = "Error analizando fechas"
    
    # Análisis de equipos
    if all(col in df.columns for col in ['home', 'away']):
        equipos = pd.concat([df['home'], df['away']]).unique()
        reporte['equipos_unicos'] = {
            'total': len(equipos),
            'lista': sorted(equipos.tolist())
        }
    
    # Análisis de ligas
    if 'liga' in df.columns:
        reporte['ligas_detectadas'] = df['liga'].value_counts().to_dict()
    
    # Análisis específico por tipo
    if info_estructura['tipo'] == 'progol_historico':
        if 'resultado' in df.columns:
            reporte['distribucion_resultados'] = df['resultado'].value_counts().to_dict()
        
        if all(col in df.columns for col in ['l_g', 'a_g']):
            reporte['estadisticas_goles'] = {
                'promedio_local': float(df['l_g'].mean()),
                'promedio_visitante': float(df['a_g'].mean()),
                'total_goles': int(df['l_g'].sum() + df['a_g'].sum())
            }
    
    return reporte

# === FUNCIÓN PRINCIPAL ===
def main():
    """
    Función principal que ejecuta todo el procesamiento dinámico
    """
    try:
        logger.info("Iniciando procesamiento dinámico de Progol")
        
        # Cargar y procesar archivo
        df, info = cargar_progol_dinamico()
        
        # Guardar archivo procesado
        output_path = guardar_progol_procesado(df, info)
        
        # Generar reporte
        reporte = generar_reporte_progol(df, info)
        
        # Guardar reporte
        reporte_path = Path("data/processed") / f"reporte_progol_{reporte['jornada_detectada']}.json"
        import json
        with open(reporte_path, 'w', encoding='utf-8') as f:
            json.dump(reporte, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Procesamiento completado")
        logger.info(f"📊 Archivo procesado: {output_path}")
        logger.info(f"📋 Reporte generado: {reporte_path}")
        
        print("\n" + "="*60)
        print("🎯 RESUMEN DEL PROCESAMIENTO")
        print("="*60)
        print(f"📁 Tipo de archivo: {reporte['tipo_archivo']}")
        print(f"📊 Registros procesados: {reporte['total_registros']}")
        print(f"📅 Jornada detectada: {reporte['jornada_detectada']}")
        if reporte['equipos_unicos']:
            print(f"⚽ Equipos únicos: {reporte['equipos_unicos']['total']}")
        if reporte['problemas']:
            print(f"⚠️ Problemas detectados: {len(reporte['problemas'])}")
            for problema in reporte['problemas']:
                print(f"   • {problema}")
        print("="*60)
        
        return True
        
    except Exception as e:
        logger.error(f"Error en procesamiento: {e}")
        print(f"\n❌ Error: {e}")
        return False

# Función de compatibilidad
def cargar_progol(path=None):
    """
    Función de compatibilidad con código existente
    """
    df, info = cargar_progol_dinamico(path)
    return df

if __name__ == "__main__":
    main()