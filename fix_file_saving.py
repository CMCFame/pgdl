#!/usr/bin/env python3
"""
Script de diagnóstico y corrección para el problema de guardado de archivos
Progol Engine - Diagnóstico de Persistencia de Archivos
"""

import os
import sys
import shutil
from pathlib import Path
import pandas as pd
import stat

def print_header(message):
    print(f"\n{'='*60}")
    print(f"🔧 {message}")
    print(f"{'='*60}")

def print_step(message):
    print(f"\n📋 {message}")

def print_success(message):
    print(f"✅ {message}")

def print_error(message):
    print(f"❌ {message}")

def print_warning(message):
    print(f"⚠️ {message}")

def diagnose_current_situation():
    """Diagnosticar la situación actual"""
    print_header("DIAGNÓSTICO INICIAL")
    
    # 1. Directorio de trabajo actual
    current_dir = Path.cwd()
    print_step(f"Directorio actual: {current_dir}")
    
    # 2. Verificar estructura de directorios esperada
    expected_dirs = ['data/raw', 'data/processed', 'streamlit_app', 'src']
    print_step("Verificando estructura de directorios:")
    
    structure_ok = True
    for dir_path in expected_dirs:
        if Path(dir_path).exists():
            print_success(f"  {dir_path} - EXISTE")
        else:
            print_error(f"  {dir_path} - NO EXISTE")
            structure_ok = False
    
    # 3. Verificar permisos de escritura
    print_step("Verificando permisos de escritura:")
    
    test_dirs = ['data', 'data/raw', 'data/processed']
    permissions_ok = True
    
    for dir_path in test_dirs:
        try:
            test_file = Path(dir_path) / "test_write.tmp"
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            
            # Intentar escribir archivo de prueba
            with open(test_file, 'w') as f:
                f.write("test")
            
            # Limpiar archivo de prueba
            test_file.unlink()
            print_success(f"  {dir_path} - ESCRITURA OK")
            
        except Exception as e:
            print_error(f"  {dir_path} - ERROR: {e}")
            permissions_ok = False
    
    # 4. Verificar archivos actualmente presentes
    print_step("Archivos actualmente en data/raw:")
    raw_dir = Path("data/raw")
    if raw_dir.exists():
        files = list(raw_dir.glob("*"))
        if files:
            for file in files:
                print(f"  📄 {file.name} ({file.stat().st_size} bytes)")
        else:
            print_warning("  Directorio vacío")
    
    print_step("Archivos actualmente en data/processed:")
    processed_dir = Path("data/processed")
    if processed_dir.exists():
        files = list(processed_dir.glob("*"))
        if files:
            for file in files:
                print(f"  📄 {file.name} ({file.stat().st_size} bytes)")
        else:
            print_warning("  Directorio vacío")
    
    return structure_ok and permissions_ok

def fix_directory_structure():
    """Crear y corregir estructura de directorios"""
    print_header("CORRECCIÓN DE ESTRUCTURA")
    
    required_dirs = [
        'data',
        'data/raw', 
        'data/processed',
        'data/dashboard',
        'data/outputs',
        'logs'
    ]
    
    print_step("Creando directorios necesarios...")
    
    for dir_path in required_dirs:
        try:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            
            # Verificar permisos y ajustar si es necesario
            if os.name != 'nt':  # Solo en Linux/Mac
                os.chmod(dir_path, 0o755)
            
            print_success(f"  {dir_path} - CREADO/VERIFICADO")
            
        except Exception as e:
            print_error(f"  {dir_path} - ERROR: {e}")

def fix_permissions():
    """Corregir permisos de archivos y directorios"""
    print_header("CORRECCIÓN DE PERMISOS")
    
    if os.name == 'nt':  # Windows
        print_warning("Sistema Windows detectado - saltando ajuste de permisos Unix")
        return
    
    print_step("Ajustando permisos en directorio data/")
    
    try:
        # Ajustar permisos del directorio data y subdirectorios
        for root, dirs, files in os.walk('data'):
            # Directorios: 755 (rwxr-xr-x)
            for dir_name in dirs:
                dir_path = Path(root) / dir_name
                os.chmod(dir_path, 0o755)
            
            # Archivos: 644 (rw-r--r--)
            for file_name in files:
                file_path = Path(root) / file_name
                os.chmod(file_path, 0o644)
        
        print_success("Permisos ajustados correctamente")
        
    except Exception as e:
        print_error(f"Error ajustando permisos: {e}")

def create_test_files():
    """Crear archivos de prueba para verificar el funcionamiento"""
    print_header("CREACIÓN DE ARCHIVOS DE PRUEBA")
    
    print_step("Creando archivos de prueba...")
    
    # Crear archivo de prueba en data/raw
    try:
        test_progol = pd.DataFrame({
            'jornada': [2287] * 14,
            'partido': range(1, 15),
            'home': [f'Equipo_{i}_Local' for i in range(1, 15)],
            'away': [f'Equipo_{i}_Visitante' for i in range(1, 15)],
            'tipo': ['Regular'] * 14
        })
        
        test_progol.to_csv('data/raw/Progol_test.csv', index=False)
        print_success("  Archivo de prueba Progol_test.csv creado")
        
    except Exception as e:
        print_error(f"  Error creando Progol_test.csv: {e}")
    
    # Crear archivo de prueba en data/processed
    try:
        test_processed = pd.DataFrame({
            'match_id': ['2287-1', '2287-2', '2287-3'],
            'p_l': [0.45, 0.32, 0.55],
            'p_e': [0.30, 0.35, 0.25],
            'p_v': [0.25, 0.33, 0.20]
        })
        
        test_processed.to_csv('data/processed/test_probabilities.csv', index=False)
        print_success("  Archivo de prueba test_probabilities.csv creado")
        
    except Exception as e:
        print_error(f"  Error creando test_probabilities.csv: {e}")

def create_fixed_save_function():
    """Crear versión corregida de la función de guardado"""
    print_header("GENERACIÓN DE FUNCIÓN CORREGIDA")
    
    fixed_function = '''
def save_uploaded_files_fixed():
    """Versión corregida de la función de guardado con manejo de errores"""
    import streamlit as st
    from pathlib import Path
    import os
    
    if 'uploaded_files' not in st.session_state:
        st.error("❌ No hay archivos para guardar")
        return False
    
    success_count = 0
    total_files = len(st.session_state.uploaded_files)
    
    # Asegurar que el directorio existe
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    for file_type, file_obj in st.session_state.uploaded_files.items():
        try:
            # Determinar nombre de archivo
            if file_type == 'progol':
                file_path = raw_dir / "Progol.csv"
            elif file_type == 'odds':
                file_path = raw_dir / "odds.csv"
            elif file_type == 'previas':
                file_path = raw_dir / "previas.pdf"
            elif file_type == 'elo':
                file_path = raw_dir / "elo.csv"
            elif file_type == 'squad':
                file_path = raw_dir / "squad_value.csv"
            else:
                st.warning(f"⚠️ Tipo de archivo desconocido: {file_type}")
                continue
            
            # Guardar archivo
            with open(file_path, 'wb') as f:
                file_content = file_obj.getvalue()
                f.write(file_content)
            
            # Verificar que se guardó correctamente
            if file_path.exists() and file_path.stat().st_size > 0:
                st.success(f"✅ {file_path.name} guardado ({file_path.stat().st_size} bytes)")
                success_count += 1
            else:
                st.error(f"❌ Error guardando {file_path.name}")
                
        except Exception as e:
            st.error(f"❌ Error guardando {file_type}: {e}")
    
    if success_count == total_files:
        st.success(f"🎉 Todos los archivos guardados correctamente ({success_count}/{total_files})")
        return True
    else:
        st.warning(f"⚠️ Solo {success_count}/{total_files} archivos guardados")
        return False
'''
    
    # Guardar función corregida en archivo
    try:
        with open('streamlit_app/save_function_fixed.py', 'w') as f:
            f.write(fixed_function)
        print_success("Función corregida guardada en streamlit_app/save_function_fixed.py")
    except Exception as e:
        print_error(f"Error guardando función corregida: {e}")

def create_pipeline_fix():
    """Crear versión corregida de las funciones del pipeline"""
    print_header("CORRECCIÓN DE FUNCIONES DEL PIPELINE")
    
    pipeline_fix = '''
def run_etl_step_fixed():
    """Versión corregida del paso ETL que persiste archivos"""
    import streamlit as st
    import pandas as pd
    import time
    from pathlib import Path
    
    with st.spinner("Ejecutando ETL..."):
        try:
            # Simular procesamiento
            time.sleep(1)
            
            # Verificar que existen archivos en data/raw
            raw_files = list(Path("data/raw").glob("*.csv"))
            if not raw_files:
                st.error("❌ No hay archivos CSV en data/raw. Carga los archivos primero.")
                return False
            
            # Crear directorio de salida
            processed_dir = Path("data/processed")
            processed_dir.mkdir(parents=True, exist_ok=True)
            
            # Procesar cada archivo encontrado
            jornada = st.session_state.get('current_jornada', '2287')
            
            for raw_file in raw_files:
                df = pd.read_csv(raw_file)
                
                # Generar archivo procesado
                processed_file = processed_dir / f"processed_{raw_file.stem}_{jornada}.csv"
                df.to_csv(processed_file, index=False)
                
                st.success(f"✅ Procesado: {processed_file.name}")
            
            # Actualizar estado
            st.session_state.pipeline_status['etl'] = True
            return True
            
        except Exception as e:
            st.error(f"❌ Error en ETL: {e}")
            return False

def verify_file_persistence():
    """Verificar que los archivos persisten después del guardado"""
    import streamlit as st
    from pathlib import Path
    import time
    
    st.subheader("🔍 Verificación de Persistencia")
    
    # Botón para verificar
    if st.button("🔍 Verificar Archivos Actuales"):
        raw_dir = Path("data/raw")
        processed_dir = Path("data/processed")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**data/raw:**")
            if raw_dir.exists():
                files = list(raw_dir.glob("*"))
                if files:
                    for file in files:
                        size = file.stat().st_size
                        modified = file.stat().st_mtime
                        st.write(f"📄 {file.name} ({size} bytes)")
                else:
                    st.warning("Directorio vacío")
            else:
                st.error("Directorio no existe")
        
        with col2:
            st.write("**data/processed:**")
            if processed_dir.exists():
                files = list(processed_dir.glob("*"))
                if files:
                    for file in files:
                        size = file.stat().st_size
                        st.write(f"📄 {file.name} ({size} bytes)")
                else:
                    st.warning("Directorio vacío")
            else:
                st.error("Directorio no existe")
'''
    
    try:
        with open('streamlit_app/pipeline_fixes.py', 'w') as f:
            f.write(pipeline_fix)
        print_success("Correcciones del pipeline guardadas en streamlit_app/pipeline_fixes.py")
    except Exception as e:
        print_error(f"Error guardando correcciones: {e}")

def show_next_steps():
    """Mostrar los siguientes pasos para implementar las correcciones"""
    print_header("SIGUIENTES PASOS")
    
    print_step("1. Aplicar las correcciones en tu aplicación Streamlit:")
    print("   - Copia el código de streamlit_app/save_function_fixed.py")
    print("   - Reemplaza la función save_uploaded_files() original")
    print("   - Integra las correcciones del pipeline")
    
    print_step("2. Comandos para ejecutar antes de iniciar Streamlit:")
    print("   cd pgdl")
    print("   python3.11 -c \"import fix_file_saving; fix_file_saving.fix_directory_structure()\"")
    print("   python3.11 -c \"import fix_file_saving; fix_file_saving.fix_permissions()\"")
    
    print_step("3. Verificar permisos manualmente (si es necesario):")
    print("   sudo chown -R $USER:$USER data/")
    print("   chmod -R 755 data/")
    
    print_step("4. Reiniciar la aplicación:")
    print("   source venv/bin/activate")
    print("   streamlit run streamlit_app/dashboard.py")
    
    print_step("5. Probar el flujo completo:")
    print("   - Cargar archivos en la interfaz")
    print("   - Usar el botón 'Guardar Archivos en Data/Raw'")
    print("   - Verificar que aparecen en el sistema de archivos")
    print("   - Ejecutar el pipeline completo")

def main():
    """Función principal de diagnóstico y corrección"""
    print_header("PROGOL ENGINE - DIAGNÓSTICO DE ARCHIVOS")
    print("Este script diagnostica y corrige problemas de guardado de archivos")
    
    # Ejecutar diagnóstico
    is_healthy = diagnose_current_situation()
    
    if not is_healthy:
        print_warning("Se encontraron problemas. Aplicando correcciones...")
        
        # Aplicar correcciones
        fix_directory_structure()
        fix_permissions()
        create_test_files()
    
    # Crear archivos de corrección
    create_fixed_save_function()
    create_pipeline_fix()
    
    # Mostrar siguientes pasos
    show_next_steps()
    
    print_header("DIAGNÓSTICO COMPLETADO")
    print_success("Revisa los archivos generados y sigue los pasos indicados")

if __name__ == "__main__":
    main()
