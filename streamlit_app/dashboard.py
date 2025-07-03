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
import base64
import plotly
import logging  # ← ESTE ERA EL QUE FALTABA

# Agregar src al path para imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from src.utils.multiple_quinielas_ui import multiple_quinielas_section
    MULTIPLE_AVAILABLE = True
except ImportError:
    MULTIPLE_AVAILABLE = False
    logging.warning("Módulo de quinielas múltiples no disponible")

# Configuración de página
st.set_page_config(
    page_title="🔢 Progol Engine v2",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_custom_css():
    """Cargar estilos personalizados mejorados"""
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #2ecc71, #27ae60);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #2ecc71;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .success-box {
        background: linear-gradient(45deg, #d4edda, #c3e6cb);
        border: 1px solid #28a745;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .warning-box {
        background: linear-gradient(45deg, #fff3cd, #ffeaa7);
        border: 1px solid #ffc107;
        color: #856404;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .error-box {
        background: linear-gradient(45deg, #f8d7da, #f5c6cb);
        border: 1px solid #dc3545;
        color: #721c24;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .match-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        margin: 0.5rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .quiniela-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        margin: 0.5rem;
    }
    .file-download {
        background: #e9ecef;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #ced4da;
        margin: 0.5rem 0;
    }
    .section-divider {
        border-top: 2px solid #2ecc71;
        margin: 2rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

def init_session_state():
    """Inicializar estado de sesión mejorado"""
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
    
    if 'partidos_config' not in st.session_state:
        st.session_state.partidos_config = {
            'regulares': 14,
            'revancha': 0,  # Por defecto sin revancha
            'total': 14
        }
    
    if 'generated_outputs' not in st.session_state:
        st.session_state.generated_outputs = {}

def create_directory_structure():
    """Crear estructura de directorios necesaria"""
    required_dirs = [
        "data/raw",
        "data/processed", 
        "data/dashboard",
        "data/reports",
        "data/json_previas",
        "data/outputs"  # Nuevo directorio para outputs
    ]
    
    for dir_path in required_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

def sidebar_navigation():
    """Navegación lateral mejorada"""
    with st.sidebar:
        st.markdown("""
        <div class="main-header">
            <h2>🔢 Progol Engine v2</h2>
            <p>Sistema Avanzado de Quinielas</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Estado del pipeline
        st.subheader("📊 Estado del Pipeline")
        for step, status in st.session_state.pipeline_status.items():
            icon = "✅" if status else "⏳"
            st.write(f"{icon} {step.title()}")
        
        st.markdown("---")
        
        # Configuración de partidos
        st.subheader("⚽ Configuración de Partidos")
        
        partidos_regulares = st.selectbox(
            "Partidos Regulares",
            options=[14],
            index=0,
            help="Progol siempre tiene 14 partidos regulares"
        )
        
        incluir_revancha = st.checkbox(
            "Incluir Revancha", 
            value=False,
            help="Agregar hasta 7 partidos de revancha opcionales"
        )
        
        partidos_revancha = 0
        if incluir_revancha:
            partidos_revancha = st.slider(
                "Partidos de Revancha",
                min_value=1,
                max_value=7,
                value=7,
                help="Número de partidos de revancha (1-7)"
            )
        
        total_partidos = partidos_regulares + partidos_revancha
        
        # Actualizar configuración
        st.session_state.partidos_config = {
            'regulares': partidos_regulares,
            'revancha': partidos_revancha,
            'total': total_partidos
        }
        
        st.info(f"**Total: {total_partidos} partidos**")
        
        st.markdown("---")
        
        # Navegación principal
        st.subheader("🧭 Navegación")
        
        # Preparar opciones de navegación
        navigation_options = [
            "🚀 Pipeline Completo",
            "📊 Análisis de Datos", 
            "🎯 Optimización Rápida",
            "📈 Métricas del Portafolio",
            "📁 Gestión de Archivos",
            "⚙️ Configuración"
        ]
        
        # Agregar quinielas múltiples si está disponible (NUEVO)
        if MULTIPLE_AVAILABLE:
            navigation_options.insert(-1, "🎯 Quinielas Múltiples")  # Antes de Configuración
        
        mode = st.radio(
            "Selecciona una sección:",
            navigation_options
        )
        
        st.markdown("---")
        
        # Información del sistema
        st.subheader("ℹ️ Sistema")
        st.info(f"""
        **Jornada Actual:** {st.session_state.current_jornada or 'No seleccionada'}
        **Archivos Cargados:** {len(st.session_state.uploaded_files)}
        **Outputs Generados:** {len(st.session_state.generated_outputs)}
        """)
        
        return mode

def upload_data_section():
    """Sección de carga de datos mejorada"""
    st.header("📤 Carga de Datos")
    
    # Información sobre los archivos requeridos
    with st.expander("ℹ️ Información sobre archivos requeridos"):
        st.markdown(f"""
        ### Archivos Necesarios para {st.session_state.partidos_config['total']} partidos:
        
        **Obligatorios:**
        - **Progol.csv**: {st.session_state.partidos_config['regulares']} partidos regulares + {st.session_state.partidos_config['revancha']} revancha
        - **odds.csv**: Momios para todos los partidos
        
        **Opcionales:**
        - **previas.pdf**: Información contextual
        - **elo.csv**: Ratings de equipos
        - **squad_value.csv**: Valores de plantilla
        """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Archivos Principales")
        
        # Progol.csv
        progol_file = st.file_uploader(
            "Progol.csv",
            type=['csv'],
            key="progol_csv",
            help=f"Archivo con {st.session_state.partidos_config['total']} partidos (14 regulares + {st.session_state.partidos_config['revancha']} revancha)"
        )
        
        if progol_file:
            st.session_state.uploaded_files['progol'] = progol_file
            
            # Validar estructura
            try:
                df = pd.read_csv(progol_file)
                expected_matches = st.session_state.partidos_config['total']
                
                if len(df) >= expected_matches:
                    st.success(f"✅ {len(df)} partidos detectados")
                    
                    # Mostrar preview
                    with st.expander("👁️ Vista previa"):
                        st.dataframe(df.head(), use_container_width=True)
                        
                        # Detectar jornada automáticamente
                        if 'jornada' in df.columns:
                            jornadas = df['jornada'].unique()
                            if len(jornadas) == 1:
                                detected_jornada = jornadas[0]
                                st.session_state.current_jornada = detected_jornada
                                st.info(f"🎯 Jornada detectada: {detected_jornada}")
                else:
                    st.warning(f"⚠️ Se esperaban {expected_matches} partidos, encontrados {len(df)}")
                    
            except Exception as e:
                st.error(f"❌ Error validando archivo: {e}")
        
        # odds.csv
        odds_file = st.file_uploader(
            "odds.csv",
            type=['csv'],
            key="odds_csv",
            help="Momios para todos los partidos"
        )
        
        if odds_file:
            st.session_state.uploaded_files['odds'] = odds_file
            
            try:
                df = pd.read_csv(odds_file)
                st.success(f"✅ {len(df)} registros de momios")
                
                with st.expander("👁️ Vista previa odds"):
                    st.dataframe(df.head(), use_container_width=True)
                    
            except Exception as e:
                st.error(f"❌ Error validando odds: {e}")
    
    with col2:
        st.subheader("📄 Archivos Opcionales")
        
        # Previas PDF
        previas_file = st.file_uploader(
            "previas.pdf",
            type=['pdf'],
            key="previas_pdf",
            help="Boletín de previas (opcional)"
        )
        
        if previas_file:
            st.session_state.uploaded_files['previas'] = previas_file
            st.success(f"✅ PDF cargado ({len(previas_file.getvalue())} bytes)")
        
        # ELO ratings
        elo_file = st.file_uploader(
            "elo.csv",
            type=['csv'],
            key="elo_csv",
            help="Ratings ELO de equipos (opcional)"
        )
        
        if elo_file:
            st.session_state.uploaded_files['elo'] = elo_file
            st.success("✅ Ratings ELO cargados")
        
        # Squad values
        squad_file = st.file_uploader(
            "squad_value.csv",
            type=['csv'],
            key="squad_csv",
            help="Valores de plantilla (opcional)"
        )
        
        if squad_file:
            st.session_state.uploaded_files['squad'] = squad_file
            st.success("✅ Valores de plantilla cargados")
    
    # Información de estado
    if st.session_state.uploaded_files:
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("📁 Archivos Cargados", len(st.session_state.uploaded_files))
        
        with col2:
            required_files = ['progol', 'odds']
            uploaded_required = [f for f in required_files if f in st.session_state.uploaded_files]
            st.metric("✅ Obligatorios", f"{len(uploaded_required)}/{len(required_files)}")
        
        with col3:
            if st.session_state.current_jornada:
                st.metric("🎯 Jornada", st.session_state.current_jornada)
            else:
                st.metric("🎯 Jornada", "No detectada")
        
        # Botón para guardar archivos
        if len(uploaded_required) == len(required_files):
            if st.button("💾 Guardar Archivos en Data/Raw", type="primary"):
                save_uploaded_files()
                st.success("✅ Archivos guardados exitosamente")
                st.rerun()
        
        return True
    
    return False

def save_uploaded_files():
    """Guardar archivos cargados en data/raw"""
    for file_type, file_obj in st.session_state.uploaded_files.items():
        if file_type == 'progol':
            file_path = "data/raw/Progol.csv"
        elif file_type == 'odds':
            file_path = "data/raw/odds.csv"
        elif file_type == 'previas':
            file_path = "data/raw/previas.pdf"
        elif file_type == 'elo':
            file_path = "data/raw/elo.csv"
        elif file_type == 'squad':
            file_path = "data/raw/squad_value.csv"
        else:
            continue
        
        # Crear directorio si no existe
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Guardar archivo
        with open(file_path, 'wb') as f:
            f.write(file_obj.getvalue())

def run_pipeline_section():
    """Sección de ejecución del pipeline mejorada"""
    st.header("🚀 Ejecución del Pipeline")
    
    if not st.session_state.current_jornada:
        st.error("❌ No se ha detectado una jornada. Carga los archivos primero.")
        return
    
    st.info(f"""
    **Configuración del Pipeline:**
    - Jornada: {st.session_state.current_jornada}
    - Partidos totales: {st.session_state.partidos_config['total']}
    - Regulares: {st.session_state.partidos_config['regulares']}
    - Revancha: {st.session_state.partidos_config['revancha']}
    """)
    
    # Pipeline steps
    pipeline_steps = [
        ("ETL", "Procesamiento de datos"),
        ("Modeling", "Cálculo de probabilidades"),
        ("Optimization", "Generación de quinielas"),
        ("Simulation", "Cálculo de métricas")
    ]
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        for step_key, step_name in pipeline_steps:
            step_lower = step_key.lower()
            status = st.session_state.pipeline_status[step_lower]
            
            with st.container():
                col_status, col_button, col_info = st.columns([1, 2, 3])
                
                with col_status:
                    icon = "✅" if status else "⏳"
                    st.write(f"{icon} **{step_name}**")
                
                with col_button:
                    button_key = f"run_{step_lower}"
                    if st.button(f"Ejecutar {step_key}", key=button_key):
                        run_pipeline_step(step_lower, step_name)
                
                with col_info:
                    if status:
                        st.success("Completado")
                    else:
                        st.info("Pendiente")
        
        st.markdown("---")
        
        # Botón para ejecutar todo
        if st.button("🚀 Ejecutar Pipeline Completo", type="primary"):
            run_complete_pipeline()
    
    with col2:
        st.subheader("📊 Estado Actual")
        
        progress = sum(st.session_state.pipeline_status.values()) / len(st.session_state.pipeline_status)
        st.progress(progress)
        st.write(f"Progreso: {progress:.0%}")
        
        if progress == 1.0:
            st.success("🎉 Pipeline completado!")
            
            # Mostrar archivos generados
            if st.session_state.generated_outputs:
                st.subheader("📁 Archivos Generados")
                for output_name, output_path in st.session_state.generated_outputs.items():
                    if Path(output_path).exists():
                        st.success(f"✅ {output_name}")
                    else:
                        st.error(f"❌ {output_name}")

def run_pipeline_step(step, step_name):
    """Ejecutar un paso individual del pipeline"""
    with st.spinner(f"Ejecutando {step_name}..."):
        try:
            if step == "etl":
                success = run_etl_step()
            elif step == "modeling":
                success = run_modeling_step()
            elif step == "optimization":
                success = run_optimization_step()
            elif step == "simulation":
                success = run_simulation_step()
            else:
                success = False
            
            if success:
                st.session_state.pipeline_status[step] = True
                st.success(f"✅ {step_name} completado")
                st.rerun()
            else:
                st.error(f"❌ Error en {step_name}")
                
        except Exception as e:
            st.error(f"❌ Error ejecutando {step_name}: {e}")

def run_complete_pipeline():
    """Ejecutar pipeline completo"""
    steps = ["etl", "modeling", "optimization", "simulation"]
    step_names = ["ETL", "Modelado", "Optimización", "Simulación"]
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, (step, step_name) in enumerate(zip(steps, step_names)):
        status_text.text(f"Ejecutando {step_name}...")
        progress_bar.progress((i) / len(steps))
        
        try:
            if step == "etl":
                success = run_etl_step()
            elif step == "modeling":
                success = run_modeling_step()
            elif step == "optimization":
                success = run_optimization_step()
            elif step == "simulation":
                success = run_simulation_step()
            
            if success:
                st.session_state.pipeline_status[step] = True
            else:
                st.error(f"❌ Error en {step_name}")
                return
                
        except Exception as e:
            st.error(f"❌ Error ejecutando {step_name}: {e}")
            return
    
    progress_bar.progress(1.0)
    status_text.text("✅ Pipeline completado!")
    st.success("🎉 ¡Pipeline ejecutado exitosamente!")
    st.rerun()

def run_etl_step():
    """Ejecutar paso ETL"""
    # Simular procesamiento ETL
    time.sleep(1)  # Simular trabajo
    
    # Crear archivos de salida simulados
    jornada = st.session_state.current_jornada
    total_partidos = st.session_state.partidos_config['total']
    
    # Generar datos de ejemplo para el número correcto de partidos
    match_data = []
    for i in range(total_partidos):
        partido_tipo = "Regular" if i < 14 else "Revancha"
        match_data.append({
            'match_id': f'{jornada}-{i+1}',
            'jornada': jornada,
            'partido': i + 1,
            'tipo': partido_tipo,
            'home': f'Equipo_{i+1}_Local',
            'away': f'Equipo_{i+1}_Visitante',
            'status': 'programado'
        })
    
    df_matches = pd.DataFrame(match_data)
    
    # Guardar en processed
    output_path = f"data/processed/matches_{jornada}.csv"
    df_matches.to_csv(output_path, index=False)
    
    st.session_state.generated_outputs['Partidos Procesados'] = output_path
    
    return True

def run_modeling_step():
    """Ejecutar paso de modelado"""
    time.sleep(2)  # Simular trabajo
    
    jornada = st.session_state.current_jornada
    total_partidos = st.session_state.partidos_config['total']
    
    # Generar probabilidades realistas
    prob_data = []
    for i in range(total_partidos):
        # Probabilidades más realistas para fútbol
        p_l = np.random.uniform(0.25, 0.55)
        p_e = np.random.uniform(0.20, 0.35)
        p_v = 1.0 - p_l - p_e
        
        # Asegurar que sumen 1
        total = p_l + p_e + p_v
        p_l /= total
        p_e /= total
        p_v /= total
        
        prob_data.append({
            'match_id': f'{jornada}-{i+1}',
            'partido': i + 1,
            'tipo': "Regular" if i < 14 else "Revancha",
            'p_final_L': p_l,
            'p_final_E': p_e,
            'p_final_V': p_v,
            'favorito': 'L' if p_l > max(p_e, p_v) else ('E' if p_e > p_v else 'V')
        })
    
    df_prob = pd.DataFrame(prob_data)
    
    # Guardar probabilidades
    output_path = f"data/processed/prob_final_{jornada}.csv"
    df_prob.to_csv(output_path, index=False)
    
    st.session_state.generated_outputs['Probabilidades'] = output_path
    
    return True

def run_optimization_step():
    """Ejecutar optimización"""
    time.sleep(3)  # Simular trabajo
    
    jornada = st.session_state.current_jornada
    total_partidos = st.session_state.partidos_config['total']
    
    # Generar portafolio de 30 quinielas
    portfolio_data = []
    
    for q in range(30):
        quiniela = {}
        quiniela['quiniela_id'] = f'Q{q+1:02d}'
        quiniela['tipo'] = 'Core' if q < 4 else 'Satélite'
        
        # Generar predicciones para cada partido
        for p in range(total_partidos):
            # Distribución realista de signos
            probs = [0.38, 0.29, 0.33]  # L, E, V históricos
            sign = np.random.choice(['L', 'E', 'V'], p=probs)
            quiniela[f'P{p+1}'] = sign
        
        # Métricas simuladas
        quiniela['l_count'] = sum(1 for v in quiniela.values() if v == 'L')
        quiniela['e_count'] = sum(1 for v in quiniela.values() if v == 'E')
        quiniela['v_count'] = sum(1 for v in quiniela.values() if v == 'V')
        
        portfolio_data.append(quiniela)
    
    df_portfolio = pd.DataFrame(portfolio_data)
    
    # Guardar portafolio
    output_path = f"data/processed/portfolio_final_{jornada}.csv"
    df_portfolio.to_csv(output_path, index=False)
    
    st.session_state.generated_outputs['Portafolio'] = output_path
    
    return True

def run_simulation_step():
    """Ejecutar simulación Monte Carlo"""
    time.sleep(2)  # Simular trabajo
    
    jornada = st.session_state.current_jornada
    
    # Generar métricas de simulación
    sim_data = []
    for q in range(30):
        sim_data.append({
            'quiniela_id': f'Q{q+1:02d}',
            'mu': np.random.normal(9.2, 0.8),  # Media de aciertos
            'sigma': np.random.uniform(0.9, 1.3),  # Desviación estándar
            'pr_10': np.random.uniform(0.25, 0.45),  # Pr[≥10]
            'pr_11': np.random.uniform(0.08, 0.18),  # Pr[≥11]
            'pr_12': np.random.uniform(0.02, 0.08),  # Pr[≥12]
            'roi_esperado': np.random.uniform(-15, 25)  # ROI %
        })
    
    df_sim = pd.DataFrame(sim_data)
    
    # Guardar métricas
    output_path = f"data/processed/simulation_metrics_{jornada}.csv"
    df_sim.to_csv(output_path, index=False)
    
    st.session_state.generated_outputs['Simulación'] = output_path
    
    return True

def analysis_section():
    """Sección de análisis mejorada"""
    st.header("📊 Análisis de Datos")
    
    if not st.session_state.current_jornada:
        st.warning("⚠️ Selecciona una jornada para analizar")
        return
    
    jornada = st.session_state.current_jornada
    
    # Verificar archivos disponibles
    available_files = check_available_files(jornada)
    
    if not available_files:
        st.error("❌ No hay archivos procesados. Ejecuta el pipeline primero.")
        return
    
    # Tabs mejorados
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Datos Cargados", 
        "🎯 Probabilidades", 
        "📈 Métricas del Portafolio",
        "📁 Archivos de Salida"
    ])
    
    with tab1:
        show_raw_data_analysis(jornada)
    
    with tab2:
        show_probabilities_analysis(jornada)
    
    with tab3:
        show_portfolio_metrics_improved(jornada)
    
    with tab4:
        show_output_files(jornada)

def check_available_files(jornada):
    """Verificar qué archivos están disponibles"""
    files_to_check = [
        f"data/processed/matches_{jornada}.csv",
        f"data/processed/prob_final_{jornada}.csv", 
        f"data/processed/portfolio_final_{jornada}.csv",
        f"data/processed/simulation_metrics_{jornada}.csv"
    ]
    
    available = {}
    for file_path in files_to_check:
        available[Path(file_path).stem] = Path(file_path).exists()
    
    return available

def show_raw_data_analysis(jornada):
    """Mostrar análisis de datos raw"""
    st.subheader(f"📋 Datos Cargados - Jornada {jornada}")
    
    matches_file = f"data/processed/matches_{jornada}.csv"
    
    if Path(matches_file).exists():
        df_matches = pd.read_csv(matches_file)
        
        # Métricas básicas
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("⚽ Total Partidos", len(df_matches))
        
        with col2:
            regulares = len(df_matches[df_matches['tipo'] == 'Regular'])
            st.metric("📋 Regulares", regulares)
        
        with col3:
            revancha = len(df_matches[df_matches['tipo'] == 'Revancha'])
            st.metric("🔄 Revancha", revancha)
        
        with col4:
            st.metric("🎯 Jornada", jornada)
        
        # Tabla de partidos
        st.subheader("📋 Lista de Partidos")
        st.dataframe(df_matches, use_container_width=True)
        
        # Distribución por tipo
        if revancha > 0:
            tipo_counts = df_matches['tipo'].value_counts()
            fig = px.pie(
                values=tipo_counts.values,
                names=tipo_counts.index,
                title="Distribución de Partidos por Tipo"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.warning("⚠️ No se encontraron datos de partidos. Ejecuta el ETL primero.")

def show_probabilities_analysis(jornada):
    """Mostrar análisis de probabilidades"""
    st.subheader(f"🎯 Análisis de Probabilidades - Jornada {jornada}")
    
    prob_file = f"data/processed/prob_final_{jornada}.csv"
    
    if Path(prob_file).exists():
        df_prob = pd.read_csv(prob_file)
        
        # Métricas de probabilidades
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_l = df_prob['p_final_L'].mean()
            st.metric("🏠 Prob. Local Promedio", f"{avg_l:.2%}")
        
        with col2:
            avg_e = df_prob['p_final_E'].mean()
            st.metric("⚖️ Prob. Empate Promedio", f"{avg_e:.2%}")
        
        with col3:
            avg_v = df_prob['p_final_V'].mean()
            st.metric("✈️ Prob. Visitante Promedio", f"{avg_v:.2%}")
        
        with col4:
            favoritos = df_prob['favorito'].value_counts()
            most_common = favoritos.index[0]
            st.metric("🎯 Resultado Más Común", f"{most_common} ({favoritos[most_common]})")
        
        # Distribución de favoritos
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(
                values=favoritos.values,
                names=favoritos.index,
                title="Distribución de Resultados Favoritos"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Gráfico de barras de probabilidades promedio
            prob_means = [avg_l, avg_e, avg_v]
            fig = px.bar(
                x=['Local', 'Empate', 'Visitante'],
                y=prob_means,
                title="Probabilidades Promedio por Resultado"
            )
            fig.update_layout(yaxis=dict(tickformat='.1%'))
            st.plotly_chart(fig, use_container_width=True)
        
        # Tabla detallada
        st.subheader("📊 Probabilidades Detalladas")
        
        # Formatear probabilidades como porcentajes
        df_display = df_prob.copy()
        for col in ['p_final_L', 'p_final_E', 'p_final_V']:
            df_display[col] = df_display[col].map(lambda x: f"{x:.1%}")
        
        st.dataframe(df_display, use_container_width=True)
        
        # Análisis por tipo (si hay revancha)
        if 'tipo' in df_prob.columns and len(df_prob['tipo'].unique()) > 1:
            st.subheader("📈 Comparación Regular vs Revancha")
            
            comparison = df_prob.groupby('tipo')[['p_final_L', 'p_final_E', 'p_final_V']].mean()
            
            fig = px.bar(
                comparison.T,
                title="Probabilidades Promedio: Regular vs Revancha",
                barmode='group'
            )
            fig.update_layout(yaxis=dict(tickformat='.1%'))
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.warning("⚠️ No se encontraron probabilidades. Ejecuta el modelado primero.")

def show_portfolio_metrics_improved(jornada):
    """Mostrar métricas del portafolio mejoradas"""
    st.subheader(f"📈 Métricas del Portafolio - Jornada {jornada}")
    
    portfolio_file = f"data/processed/portfolio_final_{jornada}.csv"
    sim_file = f"data/processed/simulation_metrics_{jornada}.csv"
    
    if not Path(portfolio_file).exists():
        st.warning("⚠️ Portafolio no encontrado. Ejecuta la optimización primero.")
        return
    
    if not Path(sim_file).exists():
        st.warning("⚠️ Métricas de simulación no encontradas. Ejecuta la simulación primero.")
        return
    
    # Cargar datos
    df_portfolio = pd.read_csv(portfolio_file)
    df_sim = pd.read_csv(sim_file)
    
    # === MÉTRICAS PRINCIPALES ===
    st.subheader("🎯 Métricas Principales")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    pr11_mean = df_sim['pr_11'].mean()
    pr10_mean = df_sim['pr_10'].mean()  
    pr12_mean = df_sim['pr_12'].mean() if 'pr_12' in df_sim.columns else 0
    mu_mean = df_sim['mu'].mean()
    sigma_mean = df_sim['sigma'].mean()
    
    with col1:
        st.metric(
            "🎯 Pr[≥11]", 
            f"{pr11_mean:.2%}",
            delta=f"{(pr11_mean - 0.081):.2%} vs histórico"
        )
    
    with col2:
        st.metric("🎯 Pr[≥10]", f"{pr10_mean:.2%}")
    
    with col3:
        st.metric("🎯 Pr[≥12]", f"{pr12_mean:.2%}")
    
    with col4:
        st.metric("🔢 μ aciertos", f"{mu_mean:.2f}")
    
    with col5:
        st.metric("📊 σ aciertos", f"{sigma_mean:.2f}")
    
    # === ANÁLISIS FINANCIERO ===
    st.subheader("💰 Análisis Financiero")
    
    col1, col2, col3, col4 = st.columns(4)
    
    costo_boleto = 15
    premio_cat2 = 90000
    n_quinielas = len(df_portfolio)
    
    costo_total = n_quinielas * costo_boleto
    ganancia_esperada = pr11_mean * premio_cat2
    roi = (ganancia_esperada / costo_total - 1) * 100
    
    with col1:
        st.metric("💸 Costo Total", f"${costo_total:,}")
    
    with col2:
        st.metric("🏆 Ganancia Esperada", f"${ganancia_esperada:,.0f}")
    
    with col3:
        st.metric(
            "💰 ROI Estimado", 
            f"{roi:.1f}%",
            delta="Positivo" if roi > 0 else "Negativo"
        )
    
    with col4:
        break_even = costo_total / premio_cat2
        st.metric("⚖️ Break-even Pr[≥11]", f"{break_even:.2%}")
    
    # === DISTRIBUCIÓN DE SIGNOS ===
    st.subheader("📊 Distribución de Signos")
    
    # Calcular distribución
    all_signs = []
    total_partidos = st.session_state.partidos_config['total']
    
    for i in range(1, total_partidos + 1):
        col_name = f'P{i}'
        if col_name in df_portfolio.columns:
            all_signs.extend(df_portfolio[col_name].tolist())
    
    if all_signs:
        signs_count = pd.Series(all_signs).value_counts()
        signs_pct = signs_count / len(all_signs) * 100
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Gráfico de distribución
            fig = px.bar(
                x=signs_count.index,
                y=signs_count.values,
                title="Distribución Total de Signos",
                color=signs_count.index,
                color_discrete_map={'L': '#3498db', 'E': '#f39c12', 'V': '#e74c3c'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Tabla de porcentajes
            dist_df = pd.DataFrame({
                'Signo': signs_count.index,
                'Cantidad': signs_count.values,
                'Porcentaje': signs_pct.values
            })
            
            st.dataframe(
                dist_df.style.format({
                    'Cantidad': '{:d}',
                    'Porcentaje': '{:.1f}%'
                }),
                use_container_width=True
            )
    
    # === ANÁLISIS POR QUINIELA ===
    st.subheader("📋 Análisis por Quiniela")
    
    # Combinar datos del portafolio con simulación
    df_combined = df_portfolio.merge(df_sim, on='quiniela_id', how='left')
    
    # Métricas por tipo de quiniela
    if 'tipo' in df_combined.columns:
        tipo_metrics = df_combined.groupby('tipo').agg({
            'pr_11': ['mean', 'std'],
            'mu': ['mean', 'std'],
            'l_count': 'mean',
            'e_count': 'mean', 
            'v_count': 'mean'
        }).round(3)
        
        st.dataframe(tipo_metrics, use_container_width=True)
    
    # === DISTRIBUCIÓN DE PR[≥11] ===
    st.subheader("📈 Distribución de Pr[≥11]")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Histograma
        fig = px.histogram(
            df_sim,
            x='pr_11',
            title="Distribución de Pr[≥11] por Quiniela",
            nbins=20
        )
        fig.update_layout(xaxis=dict(tickformat='.1%'))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Box plot por tipo si está disponible
        if 'tipo' in df_combined.columns:
            fig = px.box(
                df_combined,
                x='tipo',
                y='pr_11',
                title="Pr[≥11] por Tipo de Quiniela"
            )
            fig.update_layout(yaxis=dict(tickformat='.1%'))
            st.plotly_chart(fig, use_container_width=True)
    
    # === TABLA COMPLETA DEL PORTAFOLIO ===
    st.subheader("📋 Portafolio Completo")
    
    # Formatear datos para mostrar
    df_display = df_combined.copy()
    
    if 'pr_11' in df_display.columns:
        df_display['pr_11'] = df_display['pr_11'].map(lambda x: f"{x:.2%}")
    if 'mu' in df_display.columns:
        df_display['mu'] = df_display['mu'].map(lambda x: f"{x:.2f}")
    if 'roi_esperado' in df_display.columns:
        df_display['roi_esperado'] = df_display['roi_esperado'].map(lambda x: f"{x:.1f}%")
    
    st.dataframe(df_display, use_container_width=True)

def show_output_files(jornada):
    """Mostrar archivos de salida disponibles"""
    st.subheader(f"📁 Archivos de Salida - Jornada {jornada}")
    
    # Buscar todos los archivos relacionados con la jornada
    processed_dir = Path("data/processed")
    output_files = []
    
    if processed_dir.exists():
        for file_path in processed_dir.glob(f"*{jornada}*"):
            if file_path.is_file():
                file_size = file_path.stat().st_size
                file_modified = datetime.fromtimestamp(file_path.stat().st_mtime)
                
                output_files.append({
                    'Archivo': file_path.name,
                    'Tipo': get_file_type(file_path.name),
                    'Tamaño': format_file_size(file_size),
                    'Modificado': file_modified.strftime("%Y-%m-%d %H:%M:%S"),
                    'Ruta': str(file_path)
                })
    
    if output_files:
        df_files = pd.DataFrame(output_files)
        
        # Mostrar tabla de archivos
        st.dataframe(df_files.drop(columns=['Ruta']), use_container_width=True)
        
        st.markdown("---")
        
        # Sección de descarga
        st.subheader("⬇️ Descargar Archivos")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Descarga individual
            st.write("**Descarga Individual:**")
            
            for file_info in output_files:
                file_path = Path(file_info['Ruta'])
                
                if file_path.exists():
                    with open(file_path, 'rb') as f:
                        file_data = f.read()
                    
                    st.download_button(
                        label=f"📄 {file_info['Archivo']}",
                        data=file_data,
                        file_name=file_info['Archivo'],
                        mime=get_mime_type(file_path.suffix),
                        key=f"download_{file_info['Archivo']}"
                    )
        
        with col2:
            # Descarga en lote (ZIP)
            st.write("**Descarga en Lote:**")
            
            if st.button("📦 Crear ZIP con todos los archivos"):
                zip_data = create_zip_archive(output_files)
                
                if zip_data:
                    st.download_button(
                        label="⬇️ Descargar ZIP",
                        data=zip_data,
                        file_name=f"progol_jornada_{jornada}.zip",
                        mime="application/zip"
                    )
        
        # === VISTA PREVIA DE ARCHIVOS ===
        st.markdown("---")
        st.subheader("👁️ Vista Previa de Archivos")
        
        # Selector de archivo para vista previa
        preview_file = st.selectbox(
            "Selecciona un archivo para vista previa:",
            options=[f['Archivo'] for f in output_files],
            key="preview_selector"
        )
        
        if preview_file:
            file_path = next(f['Ruta'] for f in output_files if f['Archivo'] == preview_file)
            show_file_preview(file_path)
    
    else:
        st.warning("⚠️ No se encontraron archivos de salida. Ejecuta el pipeline primero.")

def get_file_type(filename):
    """Determinar tipo de archivo"""
    if 'portfolio' in filename:
        return "Portafolio"
    elif 'prob' in filename:
        return "Probabilidades"
    elif 'simulation' in filename:
        return "Simulación"
    elif 'matches' in filename:
        return "Partidos"
    else:
        return "Otro"

def format_file_size(size_bytes):
    """Formatear tamaño de archivo"""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024**2:
        return f"{size_bytes/1024:.1f} KB"
    else:
        return f"{size_bytes/(1024**2):.1f} MB"

def get_mime_type(extension):
    """Obtener tipo MIME por extensión"""
    mime_types = {
        '.csv': 'text/csv',
        '.json': 'application/json',
        '.pdf': 'application/pdf',
        '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    }
    return mime_types.get(extension, 'application/octet-stream')

def create_zip_archive(file_list):
    """Crear archivo ZIP con los archivos de salida"""
    zip_buffer = io.BytesIO()
    
    try:
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for file_info in file_list:
                file_path = Path(file_info['Ruta'])
                if file_path.exists():
                    zip_file.write(file_path, file_path.name)
        
        zip_buffer.seek(0)
        return zip_buffer.getvalue()
    
    except Exception as e:
        st.error(f"Error creando ZIP: {e}")
        return None

def show_file_preview(file_path):
    """Mostrar vista previa de archivo"""
    file_path = Path(file_path)
    
    try:
        if file_path.suffix == '.csv':
            df = pd.read_csv(file_path)
            st.dataframe(df.head(20), use_container_width=True)
            
            if len(df) > 20:
                st.info(f"Mostrando las primeras 20 filas de {len(df)} totales.")
        
        elif file_path.suffix == '.json':
            with open(file_path, 'r') as f:
                data = json.load(f)
            st.json(data)
        
        else:
            st.info("Vista previa no disponible para este tipo de archivo.")
    
    except Exception as e:
        st.error(f"Error mostrando vista previa: {e}")

def optimization_section():
    """Sección de optimización rápida"""
    st.header("🎯 Optimización Rápida")
    
    st.info("""
    **Optimización Rápida** permite generar un portafolio optimizado 
    saltando algunos pasos del pipeline completo.
    """)
    
    # Configuración rápida
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⚙️ Configuración")
        
        n_quinielas = st.slider("Número de quinielas", 10, 50, 30)
        estrategia = st.selectbox(
            "Estrategia de optimización",
            ["Conservadora", "Balanceada", "Agresiva"]
        )
        incluir_revancha = st.checkbox("Incluir revancha", value=False)
    
    with col2:
        st.subheader("📊 Parámetros Avanzados")
        
        if estrategia == "Conservadora":
            pr11_objetivo = st.slider("Pr[≥11] objetivo", 0.08, 0.15, 0.10)
        elif estrategia == "Balanceada":
            pr11_objetivo = st.slider("Pr[≥11] objetivo", 0.10, 0.18, 0.13)
        else:
            pr11_objetivo = st.slider("Pr[≥11] objetivo", 0.12, 0.22, 0.16)
        
        max_concentracion = st.slider("Máx. concentración por signo", 0.5, 0.8, 0.7)
    
    if st.button("🚀 Ejecutar Optimización Rápida", type="primary"):
        with st.spinner("Ejecutando optimización..."):
            # Simular optimización rápida
            time.sleep(3)
            
            # Generar resultados simulados
            generate_quick_optimization_results(n_quinielas, estrategia, incluir_revancha)
            
            st.success("✅ Optimización completada!")
            st.rerun()

def generate_quick_optimization_results(n_quinielas, estrategia, incluir_revancha):
    """Generar resultados de optimización rápida"""
    jornada = st.session_state.current_jornada or "quick"
    total_partidos = 21 if incluir_revancha else 14
    
    # Generar portafolio optimizado
    portfolio_data = []
    
    for q in range(n_quinielas):
        quiniela = {
            'quiniela_id': f'Q{q+1:02d}',
            'estrategia': estrategia,
            'tipo': 'Core' if q < 4 else 'Satélite'
        }
        
        # Generar predicciones basadas en estrategia
        signos = []
        for p in range(total_partidos):
            if estrategia == "Conservadora":
                probs = [0.45, 0.35, 0.20]  # Más locales
            elif estrategia == "Agresiva":
                probs = [0.30, 0.25, 0.45]  # Más visitantes
            else:
                probs = [0.38, 0.29, 0.33]  # Balanceada
            
            sign = np.random.choice(['L', 'E', 'V'], p=probs)
            quiniela[f'P{p+1}'] = sign
            signos.append(sign)
        
        # Calcular conteos de signos
        quiniela['l_count'] = signos.count('L')
        quiniela['e_count'] = signos.count('E')
        quiniela['v_count'] = signos.count('V')
        
        portfolio_data.append(quiniela)
    
    df_portfolio = pd.DataFrame(portfolio_data)
    
    # Guardar resultados
    output_path = f"data/processed/portfolio_final_{jornada}.csv"
    df_portfolio.to_csv(output_path, index=False)
    
    # Actualizar session state
    st.session_state.generated_outputs['Optimización Rápida'] = output_path
    st.session_state.pipeline_status['optimization'] = True
    st.session_state.current_jornada = jornada

def configuration_section():
    """Sección de configuración mejorada"""
    st.header("⚙️ Configuración del Sistema")
    
    # Configuración de archivos
    st.subheader("📁 Configuración de Archivos")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Directorios del Sistema:**")
        directories = [
            "data/raw",
            "data/processed",
            "data/dashboard", 
            "data/reports",
            "data/outputs"
        ]
        
        for directory in directories:
            if Path(directory).exists():
                st.success(f"✅ {directory}")
            else:
                st.error(f"❌ {directory}")
    
    with col2:
        st.write("**Archivos de Configuración:**")
        config_files = [
            ".env",
            "src/utils/config.py",
            "requirements.txt"
        ]
        
        for config_file in config_files:
            if Path(config_file).exists():
                st.success(f"✅ {config_file}")
            else:
                st.error(f"❌ {config_file}")
    
    # Limpieza de archivos
    st.subheader("🧹 Limpieza de Archivos")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🗑️ Limpiar Data/Processed"):
            clean_directory("data/processed")
            st.success("✅ Directorio limpiado")
    
    with col2:
        if st.button("🗑️ Limpiar Data/Dashboard"):
            clean_directory("data/dashboard")
            st.success("✅ Directorio limpiado")
    
    with col3:
        if st.button("🗑️ Limpiar Todo"):
            clean_directory("data/processed")
            clean_directory("data/dashboard")
            clean_directory("data/outputs")
            st.session_state.generated_outputs = {}
            st.success("✅ Todos los directorios limpiados")
    
    # Información del sistema
    st.subheader("ℹ️ Información del Sistema")
    
    system_info = {
        'Python': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        'Streamlit': st.__version__,
        'Pandas': pd.__version__,
        'NumPy': np.__version__,
        'Plotly': plotly.__version__,
        'Directorio de Trabajo': os.getcwd()
    }
    
    for key, value in system_info.items():
        st.info(f"**{key}**: {value}")

def clean_directory(directory):
    """Limpiar directorio"""
    dir_path = Path(directory)
    if dir_path.exists():
        for file_path in dir_path.glob("*"):
            if file_path.is_file():
                file_path.unlink()

def main():
    """Función principal mejorada"""
    load_custom_css()
    init_session_state()
    create_directory_structure()
    
    # Header principal
    st.markdown("""
    <div class="main-header">
        <h1>🔢 Progol Engine v2</h1>
        <p>Sistema Avanzado de Optimización de Quinielas con Soporte para Revancha</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Navegación
    mode = sidebar_navigation()
    
    # Contenido principal según el modo
    if mode == "🚀 Pipeline Completo":
        has_data = upload_data_section()
        if has_data or st.session_state.current_jornada:
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            run_pipeline_section()
    
    elif mode == "📊 Análisis de Datos":
        analysis_section()
    
    elif mode == "🎯 Optimización Rápida":
        optimization_section()
    
    elif mode == "📈 Métricas del Portafolio":
        if st.session_state.current_jornada:
            show_portfolio_metrics_improved(st.session_state.current_jornada)
        else:
            st.warning("⚠️ Selecciona una jornada para ver métricas")
    
    elif mode == "📁 Gestión de Archivos":
        if st.session_state.current_jornada:
            show_output_files(st.session_state.current_jornada)
        else:
            st.warning("⚠️ Selecciona una jornada para gestionar archivos")
    
    elif mode == "🎯 Quinielas Múltiples":
        if MULTIPLE_AVAILABLE:
            multiple_quinielas_section()
        else:
            st.error("🚫 Funcionalidad de Quinielas Múltiples no disponible")
            st.info("""
            📋 **Para habilitar esta función:**
            
            1. Crea el archivo `src/utils/multiple_quinielas_generator.py`
            2. Crea el archivo `src/utils/multiple_quinielas_ui.py`  
            3. Reinicia la aplicación
            
            💡 Alternativamente, usa: `python install_multiple_quinielas.py`
            """)
    
    elif mode == "⚙️ Configuración":
        configuration_section()
    
    # Footer mejorado
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**🔢 Progol Engine v2**")
        st.caption("Sistema Integral de Optimización")
    
    with col2:
        st.markdown("**⚽ Capacidades**")
        st.caption("14 Regulares + 7 Revancha = 21 Partidos")
    
    with col3:
        st.markdown("**🎯 Metodología**")
        st.caption("Core + Satélites con GRASP-Annealing")

if __name__ == "__main__":
    main()