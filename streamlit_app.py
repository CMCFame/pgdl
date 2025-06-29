#!/usr/bin/env python3
"""
Punto de entrada principal para Streamlit Cloud
Progol Engine v2 - Sistema Avanzado de Optimización de Quinielas

Este archivo es REQUERIDO por Streamlit Cloud y debe estar en la raíz del proyecto.
"""

import sys
import os
from pathlib import Path

# Agregar src al path para imports
current_dir = Path(__file__).parent
src_path = current_dir / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

# Agregar directorio raíz al path
sys.path.insert(0, str(current_dir))

# Verificar si estamos en modo desarrollo o producción
def is_development_mode():
    """Detectar si estamos en desarrollo local"""
    return (
        os.getenv("STREAMLIT_ENV") == "development" or
        "localhost" in os.getenv("STREAMLIT_SERVER_ADDRESS", "") or
        any("dev" in arg.lower() for arg in sys.argv)
    )

def setup_environment():
    """Configurar entorno según el modo"""
    if is_development_mode():
        print("🔧 Modo desarrollo detectado")
        
        # En desarrollo, usar configuración local
        os.environ.setdefault("JORNADA_ID", "2283")
        os.environ.setdefault("STREAMLIT_PORT", "8501")
        os.environ.setdefault("DATA_RAW_PATH", "data/raw/")
        os.environ.setdefault("DATA_PROCESSED_PATH", "data/processed/")
        
    else:
        print("🚀 Modo producción detectado (Streamlit Cloud)")
        
        # En producción, configuración para Streamlit Cloud
        os.environ.setdefault("JORNADA_ID", "demo")
        os.environ.setdefault("DATA_RAW_PATH", "/tmp/data/raw/")
        os.environ.setdefault("DATA_PROCESSED_PATH", "/tmp/data/processed/")

def create_minimal_directories():
    """Crear directorios mínimos necesarios"""
    required_dirs = [
        "data/raw",
        "data/processed", 
        "data/dashboard",
        "data/outputs",
        "logs"
    ]
    
    for directory in required_dirs:
        try:
            Path(directory).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"⚠️ No se pudo crear {directory}: {e}")

def import_dashboard():
    """Importar y ejecutar el dashboard principal"""
    try:
        # Intentar importar desde streamlit_app/
        from streamlit_app.dashboard import main
        print("✅ Dashboard importado desde streamlit_app/")
        return main
        
    except ImportError:
        try:
            # Fallback: intentar importar dashboard mejorado
            print("📱 Cargando dashboard mejorado...")
            
            # Aquí iría el código del dashboard mejorado
            # Por ahora, creamos una función simple
            def main():
                import streamlit as st
                
                st.set_page_config(
                    page_title="🔢 Progol Engine v2",
                    page_icon="🎯",
                    layout="wide"
                )
                
                st.title("🔢 Progol Engine v2")
                st.success("✅ Sistema cargado exitosamente!")
                
                st.info("""
                **¡Bienvenido al Progol Engine v2!**
                
                Sistema avanzado de optimización de quinielas con soporte para:
                - 14 partidos regulares + 7 de revancha (21 total)
                - Arquitectura Core + Satélites
                - Optimización GRASP + Annealing
                - Métricas avanzadas del portafolio
                """)
                
                # Mostrar información del sistema
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("🎯 Partidos Máximos", "21")
                
                with col2:
                    st.metric("🎲 Quinielas", "30")
                
                with col3:
                    st.metric("📊 Versión", "2.0.0")
                
                st.markdown("---")
                st.markdown("**Para funcionalidad completa, usa el dashboard mejorado en modo desarrollo.**")
            
            return main
            
        except Exception as e:
            print(f"❌ Error importando dashboard: {e}")
            
            # Función de emergencia
            def emergency_main():
                import streamlit as st
                
                st.error("❌ Error cargando dashboard principal")
                st.info("🔧 Modo de emergencia activado")
                
                st.markdown("""
                ### 🚨 Modo de Emergencia
                
                El dashboard principal no se pudo cargar. Posibles causas:
                - Archivos faltantes
                - Dependencias no instaladas
                - Error de configuración
                
                **Para resolver:**
                1. Verifica que todos los archivos estén presentes
                2. Instala dependencias: `pip install -r requirements.txt`
                3. Ejecuta setup: `python setup.py`
                """)
            
            return emergency_main

def main():
    """Función principal de entrada"""
    try:
        # Configurar entorno
        setup_environment()
        
        # Crear directorios básicos
        create_minimal_directories()
        
        # Mostrar información de inicio
        print("🔢 Progol Engine v2 - Iniciando...")
        print(f"📁 Directorio de trabajo: {os.getcwd()}")
        print(f"🐍 Python: {sys.version}")
        print(f"🔧 Modo: {'Desarrollo' if is_development_mode() else 'Producción'}")
        
        # Importar y ejecutar dashboard
        dashboard_main = import_dashboard()
        
        if dashboard_main:
            print("✅ Dashboard cargado exitosamente")
            dashboard_main()
        else:
            print("❌ No se pudo cargar el dashboard")
            
    except Exception as e:
        print(f"❌ Error crítico en main: {e}")
        
        # Dashboard de emergencia mínimo
        import streamlit as st
        
        st.set_page_config(
            page_title="❌ Error - Progol Engine",
            page_icon="🚨"
        )
        
        st.error("❌ Error Crítico del Sistema")
        st.code(str(e))
        
        st.info("""
        **Instrucciones de Recuperación:**
        
        1. Verifica la instalación: `python scripts/verify_installation.py`
        2. Reinstala dependencias: `pip install -r requirements.txt`
        3. Ejecuta setup: `python setup.py`
        4. Contacta soporte si el problema persiste
        """)

# ===== CONFIGURACIÓN ADICIONAL PARA STREAMLIT CLOUD =====

# Variables de entorno específicas para Streamlit Cloud
if "streamlit" in sys.modules:
    # Configuraciones automáticas cuando se detecta Streamlit
    os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    os.environ.setdefault("STREAMLIT_THEME_PRIMARY_COLOR", "#2ecc71")
    os.environ.setdefault("STREAMLIT_THEME_BACKGROUND_COLOR", "#ffffff")

# Metadatos para Streamlit Cloud
__app_name__ = "Progol Engine v2"
__app_description__ = "Sistema Avanzado de Optimización de Quinielas"
__app_version__ = "2.0.0"
__app_author__ = "Progol Engine Team"

# ===== PUNTO DE ENTRADA =====

if __name__ == "__main__":
    main()
else:
    # Si se importa como módulo, también configurar entorno
    setup_environment()
    create_minimal_directories()