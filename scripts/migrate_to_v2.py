#!/usr/bin/env python3
"""
Script de Migración - Progol Engine v2
Ayuda a migrar datos del sistema v1 al v2
"""

import pandas as pd
from pathlib import Path

def migrate_portfolio_data():
    """Migrar datos de portafolio"""
    print("🔄 Migrando datos de portafolio...")
    
    # Buscar archivos de portafolio v1
    processed_dir = Path('data/processed')
    if not processed_dir.exists():
        print("❌ Directorio data/processed no encontrado")
        return
    
    portfolio_files = list(processed_dir.glob('portfolio*.csv'))
    
    for file_path in portfolio_files:
        try:
            df = pd.read_csv(file_path)
            
            # Verificar si ya tiene formato v2
            if 'tipo' in df.columns:
                print(f"✅ {file_path.name} ya en formato v2")
                continue
            
            # Agregar columna tipo si no existe
            if 'quiniela_id' in df.columns:
                df['tipo'] = df['quiniela_id'].apply(
                    lambda x: 'Core' if x.startswith('C') else 'Satélite'
                )
            
            # Guardar versión migrada
            migrated_path = file_path.parent / f"migrated_{file_path.name}"
            df.to_csv(migrated_path, index=False)
            print(f"✅ Migrado: {file_path.name} -> {migrated_path.name}")
            
        except Exception as e:
            print(f"❌ Error migrando {file_path}: {e}")

if __name__ == "__main__":
    migrate_portfolio_data()
