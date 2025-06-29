#!/usr/bin/env python3
"""
Generador de Templates - Progol Engine v2
"""

import pandas as pd
import numpy as np

def generate_progol_template(n_partidos=21):
    """Generar template de Progol.csv"""
    data = []
    for i in range(n_partidos):
        data.append({
            'jornada': '',
            'partido': i + 1,
            'tipo': 'Regular' if i < 14 else 'Revancha',
            'local': '',
            'visitante': '',
            'goles_local': '',
            'goles_visitante': ''
        })
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    template = generate_progol_template()
    template.to_csv('progol_template.csv', index=False)
    print("✅ Template generado: progol_template.csv")
