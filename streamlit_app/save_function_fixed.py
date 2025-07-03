
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
