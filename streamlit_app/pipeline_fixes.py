
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
