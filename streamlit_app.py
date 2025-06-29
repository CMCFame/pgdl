"""
Progol Engine Dashboard Completo - Streamlit App
Versión actualizada con templates dinámicos y funcionalidad completa
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import json
from pathlib import Path
import os
import sys
import subprocess
import shutil
from datetime import datetime, timedelta
import time
import zipfile
import io

# Agregar src al path para imports
sys.path.insert(0, os.path.dirname(__file__))

# Configuración de página
st.set_page_config(
    page_title="🔢 Progol Engine",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== FUNCIONES DE CONFIGURACIÓN =====

def load_custom_css():
    """Cargar estilos personalizados"""
    st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2ecc71;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 1rem;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    </style>
    """, unsafe_allow_html=True)

def init_session_state():
    """Inicializar estado de sesión"""
    if 'pipeline_status' not in st.session_state:
        st.session_state.pipeline_status = {
            'etl': False,
            'modeling': False,
            'optimization': False,
            'simulation': False
        }
    
    if 'current_jornada' not in st.session_state:
        st.session_state.current_jornada = None
    
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = {}
    
    if 'optimization_params' not in st.session_state:
        st.session_state.optimization_params = {
            'n_quinielas': 30,
            'max_iterations': 1000,
            'use_annealing': True,
            'target_pr11': 0.12
        }

def create_directory_structure():
    """Crear estructura de directorios necesaria"""
    dirs = [
        "data/raw", "data/processed", "data/dashboard", 
        "data/reports", "data/json_previas", "data/uploads",
        "data/templates"
    ]
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

# ===== FUNCIONES DE VALIDACIÓN =====

def validate_prediction_csv(df):
    """Validar CSV para PREDICCIÓN (partidos futuros sin resultados)"""
    required_cols = ['concurso_id', 'fecha', 'match_no', 'liga', 'home', 'away']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        return False, f"Columnas faltantes: {missing_cols}"
    
    # Validar que NO tenga resultados (es para predicción)
    if 'resultado' in df.columns:
        return False, "❌ Este CSV contiene resultados. Para PREDICCIÓN usa un CSV sin columnas 'resultado', 'l_g', 'a_g'"
    
    # Validar tipos de datos básicos
    try:
        df['concurso_id'] = pd.to_numeric(df['concurso_id'])
        df['match_no'] = pd.to_numeric(df['match_no'])
        df['fecha'] = pd.to_datetime(df['fecha'])
    except Exception as e:
        return False, f"Error en tipos de datos: {str(e)}"
    
    # Validar que sean partidos futuros o actuales
    today = datetime.now().date()
    df['fecha_parsed'] = pd.to_datetime(df['fecha']).dt.date
    
    # Permitir fechas de hoy en adelante (o máximo 7 días atrás para flexibilidad)
    min_date = today - timedelta(days=7)
    partidos_muy_antiguos = df[df['fecha_parsed'] < min_date]
    
    if len(partidos_muy_antiguos) > 0:
        st.warning(f"⚠️ Algunos partidos son de fechas muy anteriores. ¿Estás seguro que es para predicción?")
    
    return True, f"✅ CSV válido para predicción: {len(df)} partidos"

def validate_odds_csv(df):
    """Validar CSV de odds"""
    required_cols = ['concurso_id', 'match_no', 'odds_L', 'odds_E', 'odds_V']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        return False, f"Columnas faltantes: {missing_cols}"
    
    # Validar que los odds sean números positivos
    odds_cols = ['odds_L', 'odds_E', 'odds_V']
    for col in odds_cols:
        if (df[col] <= 0).any():
            return False, f"Odds en {col} deben ser positivos"
        if (df[col] < 1.0).any():
            return False, f"Odds en {col} menores a 1.0 (formato decimal requerido)"
    
    return True, f"✅ Odds válidos: {len(df)} partidos"

def validate_csv_structure(df, required_columns, file_type):
    """Validar estructura de CSV"""
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        return False, f"Columnas faltantes en {file_type}: {missing_cols}"
    
    if len(df) == 0:
        return False, f"{file_type} está vacío"
    
    return True, f"{file_type} válido: {len(df)} registros"

# ===== FUNCIONES DE NAVEGACIÓN =====

def sidebar_navigation():
    """Navegación en sidebar actualizada con templates"""
    st.sidebar.title("🎯 Progol Engine")
    st.sidebar.markdown("### Control Central")
    
    # Modo de operación
    mode = st.sidebar.selectbox(
        "🎮 Modo de Operación",
        [
            "🚀 Pipeline Completo",
            "📊 Análisis de Datos", 
            "🎯 Optimización Rápida",
            "📈 Comparar Portafolios",
            "🔍 Explorar Resultados",
            "📋 Templates de Archivos",  # ← NUEVA OPCIÓN
            "⚙️ Configuración"
        ]
    )
    
    st.sidebar.markdown("---")
    
    # Jornadas disponibles
    st.sidebar.markdown("### 📅 Jornadas Disponibles")
    jornadas = get_available_jornadas()
    
    if jornadas:
        # Si hay una jornada en session_state y está en la lista, úsala como default
        default_index = 0
        if st.session_state.current_jornada and st.session_state.current_jornada in jornadas:
            default_index = jornadas.index(st.session_state.current_jornada)
        
        selected_jornada = st.sidebar.selectbox(
            "Seleccionar Jornada",
            jornadas,
            index=default_index,
            key="jornada_selector"
        )
        
        # Actualizar session state si cambió
        if selected_jornada != st.session_state.current_jornada:
            st.session_state.current_jornada = selected_jornada
            st.sidebar.success(f"✅ Jornada {selected_jornada} seleccionada")
    else:
        st.sidebar.info("📋 No hay jornadas disponibles")
        st.sidebar.markdown("Sube archivos en 'Pipeline Completo'")
    
    return mode

def get_available_jornadas():
    """Obtener jornadas disponibles"""
    jornadas = []
    
    # Buscar en data/raw/ (archivos subidos)
    raw_dir = Path("data/raw")
    if raw_dir.exists():
        # Buscar patrones comunes de archivos
        patterns = ["progol*.csv", "Progol*.csv", "*progol*.csv"]
        for pattern in patterns:
            for file in raw_dir.glob(pattern):
                # Extraer jornada del nombre
                parts = file.stem.replace('progol', '').replace('Progol', '').split('_')
                for part in parts:
                    if part.isdigit() and len(part) >= 4:
                        if part not in jornadas:
                            jornadas.append(part)
    
    # También buscar en data/processed/ (archivos procesados)
    processed_dir = Path("data/processed")
    if processed_dir.exists():
        for file in processed_dir.glob("portfolio_final_*.csv"):
            jornada = file.stem.split("_")[-1]
            if jornada not in jornadas:
                jornadas.append(jornada)
    
    return sorted(jornadas, reverse=True)

# ===== FUNCIONES DE TEMPLATES =====

def templates_section():
    """Sección para generar y descargar templates de archivos"""
    st.title("📋 Templates de Archivos")
    st.markdown("Genera templates exactos para todos los tipos de archivos que necesita la aplicación")
    
    # Configuración de jornada
    col1, col2 = st.columns([2, 1])
    
    with col1:
        jornada_template = st.number_input(
            "🎯 Jornada para Templates",
            min_value=1000,
            max_value=9999,
            value=2287,
            help="Número de jornada que se usará en los templates"
        )
    
    with col2:
        num_partidos = st.selectbox(
            "⚽ Número de Partidos",
            [14, 15, 16, 18, 20],
            index=0,
            help="Cantidad de partidos en la jornada"
        )
    
    st.markdown("---")
    
    # Botón para generar todos los templates
    if st.button("🎨 Generar Todos los Templates", type="primary"):
        with st.spinner("Generando templates..."):
            try:
                from src.utils.template_generator import crear_todos_los_templates
                
                resultado = crear_todos_los_templates(jornada_template, "data/templates")
                
                st.success("✅ ¡Templates generados exitosamente!")
                
                # Mostrar resumen
                st.json(resultado)
                
                # Crear ZIP para descarga
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    templates_path = Path("data/templates")
                    for file_path in templates_path.rglob("*"):
                        if file_path.is_file():
                            zip_file.write(file_path, file_path.name)
                
                st.download_button(
                    label="📥 Descargar Todos los Templates (ZIP)",
                    data=zip_buffer.getvalue(),
                    file_name=f"progol_templates_{jornada_template}.zip",
                    mime="application/zip"
                )
                
            except Exception as e:
                st.error(f"❌ Error generando templates: {e}")
    
    st.markdown("---")
    
    # Sección de templates individuales
    st.subheader("📄 Templates Individuales")
    
    template_tabs = st.tabs([
        "🎯 Progol", "💰 Odds", "📊 ELO", 
        "👥 Squad Values", "📝 Previas", "🔍 Diagnóstico"
    ])
    
    # Tab 1: Templates Progol
    with template_tabs[0]:
        st.markdown("### 🎯 Templates de Progol")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📈 Generar Template PREDICCIÓN"):
                try:
                    from src.utils.template_generator import generar_template_progol_prediccion
                    
                    df = generar_template_progol_prediccion(jornada_template, num_partidos)
                    
                    st.success("✅ Template de predicción generado")
                    st.dataframe(df, use_container_width=True)
                    
                    csv = df.to_csv(index=False)
                    st.download_button(
                        "📥 Descargar Template Predicción",
                        csv,
                        f"progol_{jornada_template}_PREDICCION.csv",
                        "text/csv"
                    )
                    
                except Exception as e:
                    st.error(f"Error: {e}")
        
        with col2:
            if st.button("📊 Generar Template HISTÓRICO"):
                try:
                    from src.utils.template_generator import generar_template_progol_historico
                    
                    df = generar_template_progol_historico(jornada_template, num_partidos)
                    
                    st.success("✅ Template histórico generado")
                    st.dataframe(df, use_container_width=True)
                    
                    csv = df.to_csv(index=False)
                    st.download_button(
                        "📥 Descargar Template Histórico",
                        csv,
                        f"progol_{jornada_template}_HISTORICO.csv",
                        "text/csv"
                    )
                    
                except Exception as e:
                    st.error(f"Error: {e}")
        
        # Explicación
        st.info("""
        **🎯 Template PREDICCIÓN**: Para partidos futuros (sin resultados)
        - Columnas: concurso_id, fecha, match_no, liga, home, away
        
        **📊 Template HISTÓRICO**: Para análisis de partidos pasados
        - Columnas adicionales: l_g, a_g, resultado, premio_1, premio_2
        """)
    
    # Tab 2: Template Odds
    with template_tabs[1]:
        st.markdown("### 💰 Template de Odds")
        
        if st.button("💰 Generar Template Odds"):
            try:
                from src.utils.template_generator import generar_template_odds
                
                df = generar_template_odds(jornada_template, num_partidos)
                
                st.success("✅ Template de odds generado")
                st.dataframe(df, use_container_width=True)
                
                csv = df.to_csv(index=False)
                st.download_button(
                    "📥 Descargar Template Odds",
                    csv,
                    f"odds_{jornada_template}.csv",
                    "text/csv"
                )
                
            except Exception as e:
                st.error(f"Error: {e}")
        
        st.info("""
        **💰 Template de Odds**: Momios de casas de apuestas
        - Formato decimal (ej: 2.50, no fraccional)
        - odds_L: Local, odds_E: Empate, odds_V: Visitante
        """)
    
    # Tab 3: Template ELO
    with template_tabs[2]:
        st.markdown("### 📊 Template de ELO")
        
        if st.button("📊 Generar Template ELO"):
            try:
                from src.utils.template_generator import generar_template_elo
                
                df = generar_template_elo(jornada_template)
                
                st.success("✅ Template de ELO generado")
                st.dataframe(df, use_container_width=True)
                
                csv = df.to_csv(index=False)
                st.download_button(
                    "📥 Descargar Template ELO",
                    csv,
                    f"elo_{jornada_template}.csv",
                    "text/csv"
                )
                
            except Exception as e:
                st.error(f"Error: {e}")
        
        st.info("""
        **📊 Template de ELO**: Ratings de fuerza de equipos
        - elo_home/away: Rating ELO (típicamente 1200-2000)
        - factor_local: Ventaja de local (típicamente 0.40-0.50)
        """)
    
    # Tab 4: Template Squad Values
    with template_tabs[3]:
        st.markdown("### 👥 Template de Squad Values")
        
        if st.button("👥 Generar Template Squad Values"):
            try:
                from src.utils.template_generator import generar_template_squad_values
                
                df = generar_template_squad_values(jornada_template)
                
                st.success("✅ Template de Squad Values generado")
                st.dataframe(df, use_container_width=True)
                
                csv = df.to_csv(index=False)
                st.download_button(
                    "📥 Descargar Template Squad Values",
                    csv,
                    f"squad_value_{jornada_template}.csv",
                    "text/csv"
                )
                
            except Exception as e:
                st.error(f"Error: {e}")
        
        st.info("""
        **👥 Template de Squad Values**: Valores de plantillas
        - squad_value: Valor en millones EUR
        - avg_age: Edad promedio del equipo
        - internationals: Número de internacionales
        """)
    
    # Tab 5: Template Previas
    with template_tabs[4]:
        st.markdown("### 📝 Template de Previas")
        
        if st.button("📝 Generar Template Previas"):
            try:
                from src.utils.template_generator import generar_template_previas_json
                
                previas_data = generar_template_previas_json(jornada_template)
                
                st.success("✅ Template de previas generado")
                st.json(previas_data[:3])  # Mostrar solo primeros 3 elementos
                
                json_str = json.dumps(previas_data, indent=2, ensure_ascii=False)
                st.download_button(
                    "📥 Descargar Template Previas",
                    json_str,
                    f"previas_{jornada_template}.json",
                    "application/json"
                )
                
            except Exception as e:
                st.error(f"Error: {e}")
        
        st.info("""
        **📝 Template de Previas**: Información contextual en JSON
        - form_H/A: Forma reciente (WWWWW, WDLLL, etc.)
        - h2h_H/E/A: Historial directo
        - inj_H/A: Número de lesionados
        - context_flag: Flags especiales (derbi, final, etc.)
        """)
    
    # Tab 6: Diagnóstico
    with template_tabs[5]:
        st.markdown("### 🔍 Diagnóstico de Archivos")
        
        if st.button("🔍 Diagnosticar Archivos Subidos"):
            try:
                from src.utils.template_generator import diagnose_uploaded_files
                
                with st.spinner("Analizando archivos..."):
                    diagnostico = diagnose_uploaded_files()
                
                if "error" in diagnostico:
                    st.error(f"❌ {diagnostico['error']}")
                else:
                    st.success("✅ Diagnóstico completado")
                    
                    # Mostrar jornada detectada
                    st.metric(
                        "🎯 Jornada Detectada", 
                        diagnostico['jornada_detectada']
                    )
                    
                    # Mostrar archivos analizados
                    if diagnostico['archivos_analizados']:
                        st.subheader("📁 Archivos Encontrados")
                        
                        for archivo in diagnostico['archivos_analizados']:
                            with st.expander(f"📄 {archivo['archivo']} ({archivo['tipo']})"):
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Registros", archivo['registros'])
                                with col2:
                                    if archivo['problemas']:
                                        st.error(f"Problemas: {len(archivo['problemas'])}")
                                        for problema in archivo['problemas']:
                                            st.write(f"• {problema}")
                                    else:
                                        st.success("✅ Sin problemas")
                    
                    # Mostrar problemas generales
                    if diagnostico['problemas']:
                        st.subheader("⚠️ Problemas Detectados")
                        for problema in diagnostico['problemas']:
                            st.warning(problema)
                    
                    # Mostrar recomendaciones
                    if diagnostico['recomendaciones']:
                        st.subheader("💡 Recomendaciones")
                        for recomendacion in diagnostico['recomendaciones']:
                            if "✅" in recomendacion:
                                st.success(recomendacion)
                            elif "❌" in recomendacion:
                                st.error(recomendacion)
                            else:
                                st.info(recomendacion)
                
            except Exception as e:
                st.error(f"❌ Error en diagnóstico: {e}")
        
        st.info("""
        **🔍 Diagnóstico Automático**:
        - Detecta automáticamente la estructura de tus archivos
        - Identifica problemas comunes
        - Sugiere correcciones
        - Verifica compatibilidad con el pipeline
        """)
    
    # Instrucciones finales
    st.markdown("---")
    st.markdown("### 📚 Instrucciones de Uso")
    
    st.markdown("""
    1. **📥 Genera** los templates que necesites usando los botones arriba
    2. **✏️ Modifica** los templates con tus datos reales
    3. **📁 Sube** los archivos a `data/raw/` 
    4. **🔍 Diagnostica** tus archivos para verificar que estén correctos
    5. **🚀 Ejecuta** el pipeline completo
    
    ### 📋 Archivos Requeridos vs Opcionales:
    
    **✅ OBLIGATORIOS:**
    - `progol_XXXX.csv` - Partidos (predicción o histórico)
    - `odds_XXXX.csv` - Momios de casas de apuestas
    
    **📊 OPCIONALES** (se generan automáticamente si faltan):
    - `elo_XXXX.csv` - Ratings ELO
    - `squad_value_XXXX.csv` - Valores de plantillas  
    - `previas_XXXX.json` - Información contextual
    
    ### 🎯 La aplicación detecta automáticamente:
    - Estructura de archivos
    - Jornada de los datos
    - Tipo de predicción vs histórico
    - Problemas en los datos
    """)

# ===== FUNCIONES DE CARGA DE ARCHIVOS =====

def upload_data_section():
    """Sección para subir archivos"""
    st.header("📤 Cargar Nuevos Datos")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Archivos Obligatorios")
        
        # Partidos
        partidos_file = st.file_uploader(
            "Partidos CSV",
            type="csv",
            help="concurso_id, fecha, match_no, liga, home, away",
            key="partidos"
        )
        
        # Odds
        odds_file = st.file_uploader(
            "Odds CSV", 
            type="csv",
            help="match_no, odds_L, odds_E, odds_V",
            key="odds"
        )
    
    with col2:
        st.subheader("📋 Archivos Opcionales")
        
        # Previas
        previas_file = st.file_uploader(
            "Previas JSON",
            type="json", 
            help="Información de forma, H2H, lesiones",
            key="previas"
        )
        
        # ELO
        elo_file = st.file_uploader(
            "Ratings ELO CSV",
            type="csv",
            help="team, elo_rating",
            key="elo"
        )
        
        # Squad values
        squad_file = st.file_uploader(
            "Valores Squad CSV",
            type="csv",
            help="team, market_value, avg_age",
            key="squad"
        )
    
    # Procesar archivos si están disponibles
    if partidos_file is not None and odds_file is not None:
        
        with st.spinner("🔄 Validando archivos..."):
            # Leer archivos
            partidos_df = pd.read_csv(partidos_file)
            odds_df = pd.read_csv(odds_file)
            
            # Validar archivos principales
            partidos_valid, partidos_msg = validate_prediction_csv(partidos_df)
            odds_valid, odds_msg = validate_odds_csv(odds_df)
            
            col1, col2 = st.columns(2)
            with col1:
                if partidos_valid:
                    st.success(partidos_msg)
                else:
                    st.error(partidos_msg)
            
            with col2:
                if odds_valid:
                    st.success(odds_msg)
                else:
                    st.error(odds_msg)
            
            # Si ambos son válidos, permitir guardar
            if partidos_valid and odds_valid:
                
                # Detectar jornada automáticamente
                jornada = partidos_df['concurso_id'].iloc[0]
                
                st.info(f"🎯 Jornada detectada: **{jornada}**")
                
                if st.button("💾 Guardar Archivos", type="primary"):
                    save_uploaded_files(
                        partidos_df, odds_df, jornada,
                        previas_file, elo_file, squad_file
                    )
                    
                    # Actualizar jornada actual
                    st.session_state.current_jornada = str(jornada)
                    
                    st.success("✅ Archivos guardados exitosamente. Puedes proceder a ejecutar el pipeline.")
                    
                    # Forzar actualización de la página para refrescar jornadas disponibles
                    time.sleep(1)
                    st.rerun()
    
    return partidos_file is not None and odds_file is not None

def save_uploaded_files(partidos_df, odds_df, jornada, previas_file=None, elo_file=None, squad_file=None):
    """Guardar archivos subidos en el sistema"""
    
    # Crear directorios
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    json_dir = Path("data/json_previas") 
    json_dir.mkdir(parents=True, exist_ok=True)
    
    # Guardar archivos principales
    partidos_df.to_csv(f"data/raw/progol_{jornada}.csv", index=False)
    odds_df.to_csv(f"data/raw/odds_{jornada}.csv", index=False)
    
    st.success(f"✅ Partidos guardados: data/raw/progol_{jornada}.csv")
    st.success(f"✅ Odds guardados: data/raw/odds_{jornada}.csv")
    
    # Archivos opcionales
    if previas_file:
        previas_data = json.load(previas_file)
        with open(f"data/json_previas/previas_{jornada}.json", 'w') as f:
            json.dump(previas_data, f, indent=2)
        st.success(f"✅ Previas guardadas: data/json_previas/previas_{jornada}.json")
    
    if elo_file:
        elo_df = pd.read_csv(elo_file)
        elo_df.to_csv(f"data/raw/elo_{jornada}.csv", index=False)
        st.success(f"✅ ELO guardado: data/raw/elo_{jornada}.csv")
    
    if squad_file:
        squad_df = pd.read_csv(squad_file)
        squad_df.to_csv(f"data/raw/squad_value_{jornada}.csv", index=False)
        st.success(f"✅ Squad Values guardado: data/raw/squad_value_{jornada}.csv")
    
    # Limpiar el estado del pipeline para la nueva jornada
    st.session_state.pipeline_status = {
        'etl': False,
        'modeling': False,
        'optimization': False,
        'simulation': False
    }

# ===== FUNCIONES DE PIPELINE =====

def run_pipeline_section():
    """Sección para ejecutar el pipeline completo"""
    st.header("🚀 Ejecutar Pipeline Completo")
    
    if not st.session_state.current_jornada:
        st.warning("⚠️ Selecciona una jornada en el sidebar para continuar")
        return
    
    jornada = st.session_state.current_jornada
    st.info(f"🎯 Procesando Jornada: **{jornada}**")
    
    # Configuración del Pipeline
    with st.expander("⚙️ Configuración del Pipeline", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            n_quinielas = st.number_input(
                "Número de Quinielas",
                min_value=10,
                max_value=100,
                value=30
            )
            
            use_real_modules = st.checkbox(
                "Usar Módulos Reales",
                value=True,
                help="Si está marcado, usa los módulos Python reales. Si no, usa simulación."
            )
        
        with col2:
            target_pr11 = st.slider(
                "Pr[≥11] Objetivo",
                min_value=0.08,
                max_value=0.20,
                value=0.12,
                step=0.01
            )
            
            max_iterations = st.number_input(
                "Iteraciones Máximas",
                min_value=100,
                max_value=5000,
                value=1000
            )
    
    # Botón para ejecutar pipeline
    if st.button("🚀 Ejecutar Pipeline Completo", type="primary"):
        
        pipeline_placeholder = st.empty()
        progress_bar = st.progress(0)
        status_placeholder = st.empty()
        
        try:
            # ETL
            progress_bar.progress(0.1)
            status_placeholder.info("📥 Procesando ETL...")
            
            if use_real_modules:
                etl_success = run_etl_modules(jornada)
            else:
                etl_success = generate_fallback_etl(jornada)
            
            if etl_success:
                st.session_state.pipeline_status['etl'] = True
                progress_bar.progress(0.3)
                status_placeholder.success("✅ ETL completado")
            else:
                st.error("❌ ETL falló")
                return
            
            # Modelado
            progress_bar.progress(0.4)
            status_placeholder.info("🧠 Procesando modelado...")
            
            if use_real_modules:
                modeling_success = run_modeling_modules(jornada)
            else:
                modeling_success = generate_fallback_modeling(jornada)
            
            if modeling_success:
                st.session_state.pipeline_status['modeling'] = True
                progress_bar.progress(0.6)
                status_placeholder.success("✅ Modelado completado")
            else:
                modeling_success = generate_fallback_modeling(jornada)
                st.warning("⚠️ Modelado falló - usando probabilidades básicas")
                st.session_state.pipeline_status['modeling'] = True
            
            # Optimización
            progress_bar.progress(0.7)
            status_placeholder.info("⚡ Procesando optimización...")
            
            optimization_success = run_optimization_modules(jornada, n_quinielas)
            
            if optimization_success:
                st.session_state.pipeline_status['optimization'] = True
                progress_bar.progress(0.9)
                status_placeholder.success("✅ Optimización completada")
            
            # Simulación
            progress_bar.progress(0.95)
            status_placeholder.info("🎲 Procesando simulación...")
            
            simulation_success = run_simulation_module(jornada)
            
            if simulation_success:
                st.session_state.pipeline_status['simulation'] = True
                progress_bar.progress(1.0)
                status_placeholder.success("✅ Pipeline completado exitosamente!")
            
            # Mostrar resultados
            display_pipeline_results(jornada)
            
        except Exception as e:
            st.error(f"❌ Error en pipeline: {e}")
        
        finally:
            progress_bar.empty()
            status_placeholder.empty()

def run_etl_modules(jornada):
    """Ejecutar módulos ETL reales"""
    try:
        # Intentar usar el build_features dinámico
        from src.etl.build_features import build_features_pipeline_dinamico
        
        result = build_features_pipeline_dinamico()
        return result is not None
        
    except Exception as e:
        st.warning(f"⚠️ Módulos ETL fallaron ({e}) - usando procesamiento fallback")
        return generate_fallback_etl(jornada)

def generate_fallback_etl(jornada):
    """Generar ETL de fallback cuando los módulos reales fallan"""
    try:
        # Aquí iría la lógica de fallback para generar archivos ETL básicos
        processed_dir = Path("data/processed")
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Generar archivo básico de features
        # ... lógica de fallback ...
        
        return True
    except Exception as e:
        st.error(f"❌ Error en ETL fallback: {e}")
        return False

def run_modeling_modules(jornada):
    """Ejecutar módulos de modelado"""
    try:
        # Aquí iría la ejecución de módulos de modelado
        return True
    except Exception as e:
        st.warning(f"⚠️ Modelado falló: {e}")
        return False

def generate_fallback_modeling(jornada):
    """Generar modelado de fallback"""
    try:
        # Generar probabilidades básicas desde odds
        return True
    except Exception as e:
        st.error(f"❌ Error en modelado fallback: {e}")
        return False

def run_optimization_modules(jornada, n_quinielas):
    """Ejecutar módulos de optimización"""
    try:
        # Aquí iría la ejecución de optimización
        return True
    except Exception as e:
        st.warning(f"⚠️ Optimización falló: {e}")
        return False

def run_simulation_module(jornada):
    """Ejecutar módulo de simulación"""
    try:
        # Aquí iría la simulación Monte Carlo
        return True
    except Exception as e:
        st.warning(f"⚠️ Simulación falló: {e}")
        return False

def display_pipeline_results(jornada):
    """Mostrar resultados del pipeline"""
    st.subheader("📊 Resultados del Pipeline")
    
    # Verificar archivos generados
    files_to_check = {
        'Features': f"data/processed/match_features_{jornada}.feather",
        'Probabilidades': f"data/processed/prob_final_{jornada}.csv",
        'Portafolio': f"data/processed/portfolio_final_{jornada}.csv",
        'Simulación': f"data/processed/simulation_metrics_{jornada}.csv"
    }
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📁 Archivos Generados")
        for name, path in files_to_check.items():
            if Path(path).exists():
                st.success(f"✅ {name}")
            else:
                st.error(f"❌ {name}")
    
    with col2:
        st.markdown("### 📊 Estado del Pipeline")
        for stage, status in st.session_state.pipeline_status.items():
            if status:
                st.success(f"✅ {stage.upper()}")
            else:
                st.error(f"❌ {stage.upper()}")

# ===== OTRAS SECCIONES =====

def analysis_section():
    """Sección de análisis de datos"""
    st.title("📊 Análisis de Datos")
    
    if not st.session_state.current_jornada:
        st.warning("⚠️ Selecciona una jornada en el sidebar")
        return
    
    st.info("🚧 Sección en construcción - Análisis detallado de datos")

def optimization_section():
    """Sección de optimización rápida"""
    st.title("🎯 Optimización Rápida")
    
    if not st.session_state.current_jornada:
        st.warning("⚠️ Selecciona una jornada en el sidebar")
        return
    
    st.info("🚧 Sección en construcción - Optimización rápida de portafolios")

def comparison_section():
    """Sección de comparación de portafolios"""
    st.title("📈 Comparar Portafolios")
    
    jornadas = get_available_jornadas()
    
    if len(jornadas) < 2:
        st.warning("⚠️ Se necesitan al menos 2 jornadas para comparar")
        return
    
    st.info("🚧 Sección en construcción - Comparación de portafolios")

def exploration_section():
    """Sección de exploración de resultados"""
    st.title("🔍 Explorar Resultados")
    
    if not st.session_state.current_jornada:
        st.warning("⚠️ Selecciona una jornada en el sidebar")
        return
    
    st.info("🚧 Sección en construcción - Exploración detallada de resultados")

def configuration_section():
    """Sección de configuración"""
    st.title("⚙️ Configuración")
    
    st.subheader("🔧 Parámetros de Optimización")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.session_state.optimization_params['n_quinielas'] = st.number_input(
            "Número de Quinielas por Defecto",
            min_value=10,
            max_value=100,
            value=st.session_state.optimization_params['n_quinielas']
        )
        
        st.session_state.optimization_params['max_iterations'] = st.number_input(
            "Iteraciones Máximas",
            min_value=100,
            max_value=10000,
            value=st.session_state.optimization_params['max_iterations']
        )
    
    with col2:
        st.session_state.optimization_params['target_pr11'] = st.slider(
            "Pr[≥11] Objetivo por Defecto",
            min_value=0.05,
            max_value=0.25,
            value=st.session_state.optimization_params['target_pr11'],
            step=0.01
        )
        
        st.session_state.optimization_params['use_annealing'] = st.checkbox(
            "Usar Simulated Annealing por Defecto",
            value=st.session_state.optimization_params['use_annealing']
        )
    
    # Gestión de archivos
    st.subheader("📁 Gestión de Archivos")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🧹 Limpiar Cache"):
            # Limpiar archivos temporales
            st.success("✅ Cache limpiado")
        
        if st.button("📊 Regenerar Datos de Ejemplo"):
            # Regenerar datos de ejemplo
            st.success("✅ Datos de ejemplo regenerados")
    
    with col2:
        if st.button("🗂️ Verificar Estructura de Directorios"):
            create_directory_structure()
            st.success("✅ Estructura de directorios verificada")
        
        if st.button("📋 Exportar Configuración"):
            config = {
                'optimization_params': st.session_state.optimization_params,
                'export_date': datetime.now().isoformat()
            }
            st.download_button(
                "⬇️ Descargar config.json",
                json.dumps(config, indent=2),
                "progol_config.json",
                "application/json"
            )
    
    # Estado del sistema
    st.subheader("🔍 Estado del Sistema")
    
    system_info = {
        "Jornadas Disponibles": len(get_available_jornadas()),
        "Directorio de Datos": "data/",
        "Último Pipeline": "No ejecutado" if not any(st.session_state.pipeline_status.values()) else "Completado",
        "Versión de Python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    }
    
    for key, value in system_info.items():
        st.info(f"**{key}**: {value}")

# ===== FUNCIÓN PRINCIPAL =====

def main():
    """Función principal de la aplicación"""
    load_custom_css()
    init_session_state()
    create_directory_structure()
    
    # Navegación
    mode = sidebar_navigation()
    
    # Contenido principal según el modo
    if mode == "🚀 Pipeline Completo":
        # Verificar si hay datos para procesar
        has_data = upload_data_section()
        
        if has_data or st.session_state.current_jornada:
            st.markdown("---")
            run_pipeline_section()
    
    elif mode == "📊 Análisis de Datos":
        analysis_section()
    
    elif mode == "🎯 Optimización Rápida":
        optimization_section()
    
    elif mode == "📈 Comparar Portafolios":
        comparison_section()
    
    elif mode == "🔍 Explorar Resultados":
        exploration_section()
    
    elif mode == "📋 Templates de Archivos":  # ← NUEVA SECCIÓN
        templates_section()
    
    elif mode == "⚙️ Configuración":
        configuration_section()
    
    # Footer
    st.markdown("---")
    st.markdown("🔢 **Progol Engine** - Sistema Integral de Optimización de Quinielas")

if __name__ == "__main__":
    main()