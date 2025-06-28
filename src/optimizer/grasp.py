# src/optimizer/grasp.py - VERSION CORREGIDA
import pandas as pd
import numpy as np
import random
import os
from pathlib import Path
from itertools import combinations

def get_dynamic_jornada():
    """Obtener jornada de forma dinámica"""
    if 'JORNADA_ID' in os.environ:
        return os.environ['JORNADA_ID']
    
    # Buscar en archivos procesados
    processed_path = Path("data/processed")
    if processed_path.exists():
        # Buscar archivos con patrón jornada
        for file in processed_path.glob("*_*.csv"):
            parts = file.stem.split('_')
            for part in parts:
                if part.isdigit() and len(part) == 4:  # Formato jornada típico
                    return part
    
    return "2287"  # Fallback actual

def load_dynamic_files(jornada_id):
    """Cargar archivos necesarios de forma dinámica"""
    processed_path = Path("data/processed")
    
    # Archivos a buscar
    files_to_load = {
        'core': None,
        'satellite': None, 
        'probabilities': None
    }
    
    # Patrones de búsqueda para cada archivo
    patterns = {
        'core': [
            f"core_quinielas_{jornada_id}.csv",
            f"core_{jornada_id}.csv",
            "core_quinielas_latest.csv",
            "core_latest.csv"
        ],
        'satellite': [
            f"satellite_quinielas_{jornada_id}.csv",
            f"satellites_{jornada_id}.csv", 
            "satellite_quinielas_latest.csv",
            "satellites_latest.csv"
        ],
        'probabilities': [
            f"prob_draw_adjusted_{jornada_id}.csv",
            f"probabilidades_finales_{jornada_id}.csv",
            f"match_features_{jornada_id}.csv",
            "probabilities_latest.csv"
        ]
    }
    
    # Buscar cada archivo
    for file_type, pattern_list in patterns.items():
        for pattern in pattern_list:
            file_path = processed_path / pattern
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path)
                    files_to_load[file_type] = df
                    print(f"✅ {file_type.title()} cargado desde: {pattern}")
                    break
                except Exception as e:
                    print(f"⚠️ Error cargando {pattern}: {e}")
                    continue
    
    return files_to_load

def create_fallback_data(jornada_id):
    """Crear datos fallback si no existen archivos"""
    print("🔧 Generando datos fallback para GRASP...")
    
    # Datos mínimos para funcionar
    n_partidos = 14
    
    # Core quinielas básicas
    core_data = []
    for i in range(4):  # 4 quinielas core
        quiniela = []
        for j in range(n_partidos):
            # Generar con distribución típica (más locales, menos empates)
            rand = np.random.random()
            if rand < 0.45:
                resultado = 'L'
            elif rand < 0.70:
                resultado = 'V'
            else:
                resultado = 'E'
            quiniela.append(resultado)
        
        core_data.append({
            'quiniela_id': f'Core-{i+1}',
            **{f'P{j+1}': quiniela[j] for j in range(n_partidos)}
        })
    
    df_core = pd.DataFrame(core_data)
    
    # Satélites básicos (pares con inversiones)
    sat_data = []
    base_quiniela = core_data[0]  # Usar primer core como base
    
    for i in range(13):  # 13 pares = 26 satélites
        # Par 1: igual a base
        sat_1 = {
            'quiniela_id': f'Sat-{2*i+1}',
            **{f'P{j+1}': base_quiniela[f'P{j+1}'] for j in range(n_partidos)}
        }
        
        # Par 2: invertir en partido aleatorio
        sat_2 = sat_1.copy()
        sat_2['quiniela_id'] = f'Sat-{2*i+2}'
        
        # Invertir un partido divisor
        partido_a_invertir = i % n_partidos  # Rotar partidos
        col_name = f'P{partido_a_invertir+1}'
        
        if sat_2[col_name] == 'L':
            sat_2[col_name] = 'V'
        elif sat_2[col_name] == 'V':
            sat_2[col_name] = 'L'
        # E se mantiene
        
        sat_data.extend([sat_1, sat_2])
    
    df_sat = pd.DataFrame(sat_data)
    
    # Probabilidades básicas
    prob_data = []
    for i in range(n_partidos):
        # Distribución típica de probabilidades
        p_l = np.random.uniform(0.25, 0.55)
        p_v = np.random.uniform(0.25, 0.55)
        p_e = 1.0 - p_l - p_v
        
        # Normalizar si suma excede 1
        if p_e < 0.15:
            total = p_l + p_v + 0.15
            p_l = p_l / total * 0.85
            p_v = p_v / total * 0.85
            p_e = 0.15
        
        prob_data.append({
            'match_no': i + 1,
            'p_final_L': p_l,
            'p_final_E': p_e,
            'p_final_V': p_v
        })
    
    df_prob = pd.DataFrame(prob_data)
    
    return df_core, df_sat, df_prob

def estimar_pr11(prob_vector, n_samples=10000):
    """Estimar Pr[≥11 aciertos] con simulación Monte Carlo"""
    if len(prob_vector) != 14:
        print(f"⚠️ Vector de probabilidades incorrecto: {len(prob_vector)} != 14")
        return 0.01
    
    # Simulación vectorizada
    aciertos = np.random.binomial(1, prob_vector, size=(n_samples, 14))
    hits = aciertos.sum(axis=1)
    pr11 = (hits >= 11).mean()
    
    return max(0.001, pr11)  # Mínimo realista

def generar_candidatos_inteligentes(base_quiniela, existing_quinielas, n_candidatos=300):
    """Generar candidatos con diversificación inteligente"""
    candidatos = []
    signos = ['L', 'E', 'V']
    
    # Analizar distribución actual
    distribuciones = {'L': [], 'E': [], 'V': []}
    for quiniela in existing_quinielas:
        for i, signo in enumerate(quiniela):
            distribuciones[signo].append(i)
    
    # Contar frecuencias por posición
    freq_por_posicion = []
    for i in range(14):
        freq = {'L': 0, 'E': 0, 'V': 0}
        for quiniela in existing_quinielas:
            freq[quiniela[i]] += 1
        freq_por_posicion.append(freq)
    
    for _ in range(n_candidatos):
        nueva_quiniela = base_quiniela.copy()
        
        # Número de cambios (más cambios = más diversidad)
        n_cambios = np.random.randint(4, 8)
        posiciones_cambio = np.random.choice(14, n_cambios, replace=False)
        
        for pos in posiciones_cambio:
            # Favorecer signos menos frecuentes en esta posición
            freq_pos = freq_por_posicion[pos]
            total_freq = sum(freq_pos.values())
            
            if total_freq == 0:
                # Si no hay frecuencias, distribución uniforme
                nueva_quiniela[pos] = np.random.choice(signos)
            else:
                # Inversely weight by frequency (menos frecuente = más probable)
                weights = []
                for signo in signos:
                    if freq_pos[signo] == 0:
                        weights.append(1.0)
                    else:
                        weights.append(1.0 / freq_pos[signo])
                
                # Normalizar
                total_weight = sum(weights)
                probs = [w / total_weight for w in weights]
                
                nueva_quiniela[pos] = np.random.choice(signos, p=probs)
        
        candidatos.append(nueva_quiniela)
    
    return candidatos

def grasp_portfolio_optimized(df_core, df_sat, df_prob, alpha=0.15, n_target=30):
    """Algoritmo GRASP optimizado para completar portafolio"""
    
    print(f"🚀 Iniciando GRASP optimizado hacia {n_target} quinielas")
    
    # Validar DataFrames
    if df_core is None or len(df_core) == 0:
        print("⚠️ No hay quinielas core - usando fallback")
        df_core, df_sat, df_prob = create_fallback_data(get_dynamic_jornada())
    
    if df_prob is None or len(df_prob) == 0:
        print("⚠️ No hay probabilidades - usando fallback") 
        _, _, df_prob = create_fallback_data(get_dynamic_jornada())
    
    # Extraer quinielas existentes
    portfolio_existente = []
    portfolio_ids = []
    
    # Agregar cores
    for _, row in df_core.iterrows():
        quiniela = [row[f'P{i+1}'] for i in range(14)]
        portfolio_existente.append(quiniela)
        portfolio_ids.append(row.get('quiniela_id', f'Core-{len(portfolio_ids)+1}'))
    
    # Agregar satélites si existen
    if df_sat is not None and len(df_sat) > 0:
        for _, row in df_sat.iterrows():
            if len(portfolio_existente) >= n_target:
                break
            quiniela = [row[f'P{i+1}'] for i in range(14)]
            portfolio_existente.append(quiniela)
            portfolio_ids.append(row.get('quiniela_id', f'Sat-{len(portfolio_ids)+1}'))
    
    print(f"📊 Portfolio inicial: {len(portfolio_existente)} quinielas")
    
    # Completar con GRASP hasta n_target
    base_quiniela = portfolio_existente[0] if portfolio_existente else ['L'] * 14
    
    iteracion = 0
    max_iteraciones = min(200, (n_target - len(portfolio_existente)) * 20)
    
    while len(portfolio_existente) < n_target and iteracion < max_iteraciones:
        # Generar candidatos con diversificación
        candidatos = generar_candidatos_inteligentes(
            base_quiniela, 
            portfolio_existente, 
            n_candidatos=300
        )
        
        # Evaluar beneficio marginal de cada candidato
        beneficios = []
        for candidato in candidatos:
            # Calcular vector de probabilidades de acierto
            prob_acierto = []
            for i, signo in enumerate(candidato):
                if i < len(df_prob):
                    prob = df_prob.iloc[i].get(f'p_final_{signo}', 0.33)
                else:
                    prob = 0.33  # Fallback
                prob_acierto.append(prob)
            
            # Estimar Pr[≥11]
            pr11 = estimar_pr11(np.array(prob_acierto))
            beneficios.append((candidato, pr11))
        
        if not beneficios:
            print("⚠️ No se generaron candidatos válidos")
            break
        
        # Ordenar por beneficio
        beneficios.sort(key=lambda x: -x[1])
        
        # Selección greedy con aleatorización (GRASP)
        top_k = max(1, int(len(beneficios) * alpha))
        candidato_elegido, beneficio = random.choice(beneficios[:top_k])
        
        # Agregar al portfolio
        portfolio_existente.append(candidato_elegido)
        portfolio_ids.append(f'GRASP-{len(portfolio_ids) - len(df_core) - (len(df_sat) if df_sat is not None else 0) + 1}')
        
        iteracion += 1
        
        if iteracion % 10 == 0:
            print(f"  Iteración {iteracion}: {len(portfolio_existente)} quinielas, mejor Pr[≥11]={beneficio:.4f}")
    
    print(f"✅ GRASP completado: {len(portfolio_existente)} quinielas generadas en {iteracion} iteraciones")
    
    # Crear DataFrame final
    portfolio_data = []
    for i, (quiniela, qid) in enumerate(zip(portfolio_existente, portfolio_ids)):
        row = {'quiniela_id': qid}
        for j, signo in enumerate(quiniela):
            row[f'P{j+1}'] = signo
        portfolio_data.append(row)
    
    df_portfolio = pd.DataFrame(portfolio_data)
    return df_portfolio

def exportar_grasp(df_portfolio, jornada_id):
    """Exportar resultado de GRASP"""
    
    processed_path = Path("data/processed")
    processed_path.mkdir(parents=True, exist_ok=True)
    
    # Archivo principal
    output_file = processed_path / f"portfolio_grasp_{jornada_id}.csv"
    df_portfolio.to_csv(output_file, index=False)
    print(f"✅ Portfolio GRASP exportado: {output_file}")
    
    # Crear también copias con nombres que otros módulos esperan
    legacy_files = [
        f"portfolio_preanneal_{jornada_id}.csv",
        "portfolio_latest.csv",
        "quinielas_finales.csv"
    ]
    
    for legacy_name in legacy_files:
        legacy_path = processed_path / legacy_name
        df_portfolio.to_csv(legacy_path, index=False)
    
    return output_file

def main():
    """Función principal del módulo GRASP"""
    
    try:
        # Obtener jornada dinámica
        jornada_id = get_dynamic_jornada()
        print(f"🎯 Ejecutando GRASP para jornada: {jornada_id}")
        
        # Cargar archivos dinámicamente
        files = load_dynamic_files(jornada_id)
        
        df_core = files['core']
        df_sat = files['satellite'] 
        df_prob = files['probabilities']
        
        # Si no hay datos, crear fallback
        if df_core is None:
            print("⚠️ No se encontraron datos - generando fallback completo")
            df_core, df_sat, df_prob = create_fallback_data(jornada_id)
        
        # Ejecutar GRASP
        df_portfolio = grasp_portfolio_optimized(df_core, df_sat, df_prob)
        
        # Exportar
        output_file = exportar_grasp(df_portfolio, jornada_id)
        
        # Estadísticas finales
        print(f"\n📊 RESUMEN GRASP:")
        print(f"  🎯 Jornada: {jornada_id}")
        print(f"  📈 Quinielas finales: {len(df_portfolio)}")
        print(f"  📁 Archivo: {output_file}")
        
        # Distribución de signos
        signos_counts = {'L': 0, 'E': 0, 'V': 0}
        for col in [f'P{i+1}' for i in range(14)]:
            if col in df_portfolio.columns:
                counts = df_portfolio[col].value_counts()
                for signo in signos_counts:
                    if signo in counts:
                        signos_counts[signo] += counts[signo]
        
        total_signos = sum(signos_counts.values())
        print(f"  📊 Distribución: L={signos_counts['L']/total_signos:.1%}, E={signos_counts['E']/total_signos:.1%}, V={signos_counts['V']/total_signos:.1%}")
        
        return df_portfolio
        
    except Exception as e:
        print(f"❌ Error en GRASP: {e}")
        raise

if __name__ == "__main__":
    main()