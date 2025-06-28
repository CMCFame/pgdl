import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

def crear_datos_ejemplo():
    """Crea datos de ejemplo con el formato correcto"""
    
    try:
        # Crear directorio si no existe
        os.makedirs("data/processed", exist_ok=True)
        
        st.info("🎯 Creando datos de ejemplo con formato correcto...")
        
        # 1. Portfolio de ejemplo (30 quinielas)
        equipos = ['América', 'Cruz Azul', 'Chivas', 'Pumas', 'Tigres', 'Monterrey', 'Santos', 'Toluca']
        resultados = ['L', 'E', 'V']
        
        portfolio_data = []
        for i in range(30):
            quiniela = [np.random.choice(resultados) for _ in range(14)]
            portfolio_data.append({
                'quiniela_id': f'Q{i+1}',
                **{f'M{j+1}': quiniela[j] for j in range(14)}
            })
        
        portfolio_df = pd.DataFrame(portfolio_data)
        portfolio_df.to_csv("data/processed/portfolio_final_2283.csv", index=False)
        
        # 2. Probabilidades de ejemplo
        prob_data = []
        for i in range(14):
            # Probabilidades realistas que sumen 1
            p_l = np.random.uniform(0.25, 0.55)
            p_e = np.random.uniform(0.25, 0.35)
            p_v = 1.0 - p_l - p_e
            
            prob_data.append({
                'match_id': f'2283-{i+1}',
                'p_final_L': p_l,
                'p_final_E': p_e,
                'p_final_V': p_v
            })
        
        prob_df = pd.DataFrame(prob_data)
        prob_df.to_csv("data/processed/prob_draw_adjusted_2283.csv", index=False)
        
        st.success("✅ Datos de ejemplo creados correctamente")
        
        # Mostrar muestra de los datos creados
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 Portfolio (muestra)")
            st.dataframe(portfolio_df.head())
            
        with col2:
            st.subheader("📊 Probabilidades (muestra)")
            st.dataframe(prob_df.head())
        
        return True
        
    except Exception as e:
        st.error(f"❌ Error creando datos de ejemplo: {e}")
        return False

def verificar_y_corregir_datos():
    """Verifica los archivos CSV y los corrige si tienen problemas"""
    
    # Rutas de archivos
    portfolio_path = "data/processed/portfolio_final_2283.csv"
    prob_path = "data/processed/prob_draw_adjusted_2283.csv"
    sim_path = "data/processed/simulation_metrics_2283.csv"
    
    archivos_faltantes = []
    
    # Verificar archivos principales
    if not os.path.exists(portfolio_path):
        archivos_faltantes.append("portfolio_final_2283.csv")
    if not os.path.exists(prob_path):
        archivos_faltantes.append("prob_draw_adjusted_2283.csv")
    
    if archivos_faltantes:
        st.error(f"❌ Archivos faltantes: {', '.join(archivos_faltantes)}")
        
        # Ofrecer crear datos de ejemplo
        if st.button("🎯 Crear Datos de Ejemplo"):
            with st.spinner("Creando datos de ejemplo..."):
                if crear_datos_ejemplo():
                    st.success("✅ Datos de ejemplo listos - actualiza la página")
                    return True
        
        st.info("O ejecuta el pipeline completo para generar los archivos reales.")
        return False
    
    # Verificar formato de archivos
    try:
        portfolio_df = pd.read_csv(portfolio_path)
        prob_df = pd.read_csv(prob_path)
        
        st.info(f"📄 Portfolio: {len(portfolio_df)} quinielas con columnas: {list(portfolio_df.columns)}")
        st.info(f"📄 Probabilidades: {len(prob_df)} partidos con columnas: {list(prob_df.columns)}")
        
        # Verificar si el formato de probabilidades es correcto
        prob_cols = detectar_formato_probabilidades(prob_df)
        
        if prob_cols is None:
            st.warning("⚠️ Formato de probabilidades no reconocido")
            if st.button("🔄 Usar Datos de Ejemplo"):
                with st.spinner("Reemplazando con datos de ejemplo..."):
                    if crear_datos_ejemplo():
                        st.success("✅ Datos de ejemplo listos - actualiza la página")
                        return True
            return False
        
    except Exception as e:
        st.error(f"❌ Error leyendo archivos: {e}")
        return False
    
    # Cargar y verificar el archivo de simulación
    if os.path.exists(sim_path):
        try:
            sim_df = pd.read_csv(sim_path)
            st.info(f"📄 Archivo de simulación encontrado con columnas: {list(sim_df.columns)}")
            
            # Verificar columnas requeridas
            columnas_requeridas = ['quiniela_id', 'mu', 'sigma', 'pr_10', 'pr_11']
            columnas_faltantes = [col for col in columnas_requeridas if col not in sim_df.columns]
            
            if columnas_faltantes:
                st.warning(f"⚠️ Columnas faltantes en simulation_metrics: {columnas_faltantes}")
                st.info("🔧 Regenerando archivo de simulación con columnas correctas...")
                return regenerar_simulacion()
            else:
                st.success("✅ Archivo de simulación tiene todas las columnas necesarias")
                return True
                
        except Exception as e:
            st.error(f"❌ Error leyendo simulation_metrics: {e}")
            st.info("🔧 Regenerando archivo de simulación...")
            return regenerar_simulacion()
    else:
        st.warning("⚠️ Archivo simulation_metrics_2283.csv no encontrado")
        st.info("🔧 Generando archivo de simulación...")
        return regenerar_simulacion()

def detectar_formato_probabilidades(prob_df):
    """Detecta el formato de las columnas de probabilidades"""
    columnas = prob_df.columns.tolist()
    st.info(f"🔍 Columnas encontradas en prob_draw: {columnas}")
    
    # Formatos posibles
    formatos = {
        'p_final': ['p_final_L', 'p_final_E', 'p_final_V'],
        'prob': ['prob_L', 'prob_E', 'prob_V'],
        'p': ['p_L', 'p_E', 'p_V'],
        'minusculo': ['l', 'e', 'v'],
        'mayuscula': ['L', 'E', 'V']
    }
    
    # Detectar formato
    for nombre, cols in formatos.items():
        if all(col in columnas for col in cols):
            st.success(f"✅ Formato detectado: {nombre} - {cols}")
            return cols
    
    # Si no se encuentra un formato estándar, usar probabilidades uniformes
    st.warning("⚠️ No se encontraron columnas de probabilidades estándar")
    st.info("📝 Usando probabilidades uniformes (33.3% cada resultado)")
    return None

def regenerar_simulacion():
    """Regenera el archivo de métricas de simulación con las columnas correctas"""
    
    try:
        # Cargar datos necesarios
        portfolio_df = pd.read_csv("data/processed/portfolio_final_2283.csv")
        prob_df = pd.read_csv("data/processed/prob_draw_adjusted_2283.csv")
        
        st.info("📊 Calculando métricas de simulación...")
        
        # Detectar formato de probabilidades
        prob_cols = detectar_formato_probabilidades(prob_df)
        
        # Generar métricas para cada quiniela
        sim_data = []
        
        for _, quiniela_row in portfolio_df.iterrows():
            quiniela_id = quiniela_row['quiniela_id']
            quiniela = quiniela_row.drop('quiniela_id').tolist()
            
            # Calcular probabilidades de acierto para cada partido
            hit_probs = []
            for match_idx, resultado in enumerate(quiniela):
                if match_idx < len(prob_df) and prob_cols is not None:
                    prob_row = prob_df.iloc[match_idx]
                    
                    # Mapear resultado a índice de columna
                    resultado_map = {'L': 0, 'E': 1, 'V': 2}
                    if resultado in resultado_map:
                        col_idx = resultado_map[resultado]
                        hit_prob = prob_row[prob_cols[col_idx]]
                    else:
                        hit_prob = 0.33  # Valor por defecto
                    
                    hit_probs.append(hit_prob)
                else:
                    # Si no hay probabilidades específicas, usar valor uniforme
                    hit_probs.append(0.33)
            
            # Asegurar que tenemos exactamente 14 probabilidades
            while len(hit_probs) < 14:
                hit_probs.append(0.33)
            hit_probs = hit_probs[:14]  # Tomar solo las primeras 14
            
            # Calcular estadísticas usando distribución binomial
            mu = sum(hit_probs)  # Esperanza
            sigma = np.sqrt(sum([p * (1-p) for p in hit_probs]))  # Desviación estándar
            
            # Calcular probabilidades usando aproximación normal
            try:
                from scipy.stats import norm
                pr_10 = 1 - norm.cdf(9.5, mu, sigma)  # P(X >= 10)
                pr_11 = 1 - norm.cdf(10.5, mu, sigma)  # P(X >= 11)
            except ImportError:
                # Si scipy no está disponible, usar aproximaciones simples
                pr_10 = max(0, min(1, (mu - 10) / 14))
                pr_11 = max(0, min(1, (mu - 11) / 14))
            
            sim_data.append({
                'quiniela_id': quiniela_id,
                'mu': mu,
                'sigma': sigma,
                'pr_10': pr_10,
                'pr_11': pr_11
            })
        
        # Crear DataFrame y guardar
        sim_df = pd.DataFrame(sim_data)
        
        # Asegurar que el directorio existe
        os.makedirs("data/processed", exist_ok=True)
        
        # Guardar archivo
        sim_df.to_csv("data/processed/simulation_metrics_2283.csv", index=False)
        
        st.success(f"✅ Archivo de simulación regenerado con {len(sim_df)} quinielas")
        st.dataframe(sim_df.head())
        
        # Mostrar estadísticas de verificación
        st.info(f"""
        **📊 Resumen de métricas calculadas:**
        - μ promedio: {sim_df['mu'].mean():.2f}
        - σ promedio: {sim_df['sigma'].mean():.2f}
        - Pr[≥11] promedio: {sim_df['pr_11'].mean():.2%}
        - Pr[≥10] promedio: {sim_df['pr_10'].mean():.2%}
        """)
        
        return True
        
    except FileNotFoundError as e:
        st.error(f"❌ Archivo requerido no encontrado: {e}")
        st.info("Asegúrate de tener los archivos portfolio_final_2283.csv y prob_draw_adjusted_2283.csv")
        return False
    except Exception as e:
        st.error(f"❌ Error regenerando simulación: {e}")
        st.exception(e)  # Mostrar detalles del error
        return False

def cargar_datos_seguros():
    """Carga datos con manejo seguro de errores"""
    try:
        port = pd.read_csv("data/processed/portfolio_final_2283.csv")
        prob = pd.read_csv("data/processed/prob_draw_adjusted_2283.csv")
        sim = pd.read_csv("data/processed/simulation_metrics_2283.csv")
        return port, prob, sim
    except Exception as e:
        st.error(f"Error cargando datos: {e}")
        return None, None, None

def mostrar_quinielas(port):
    """Muestra el portafolio de quinielas"""
    if port is not None:
        st.subheader("📋 Quinielas Finales")
        st.dataframe(port, use_container_width=True)
    else:
        st.error("No se pudo cargar el portafolio")

def mostrar_distribucion_signos(port):
    """Muestra la distribución de signos"""
    if port is not None:
        st.subheader("📊 Distribución Global de Signos")
        
        # Obtener todos los signos (excluyendo la columna quiniela_id)
        signos = port.drop(columns="quiniela_id").values.flatten()
        distrib = pd.Series(signos).value_counts(normalize=True).reindex(['L','E','V'], fill_value=0)

        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        bars = ax.bar(distrib.index, distrib.values, color=colors)
        
        # Agregar etiquetas con porcentajes
        for bar, pct in zip(bars, distrib.values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{pct:.1%}', ha='center', va='bottom')
        
        ax.set_ylim(0, max(distrib.values) * 1.2)
        ax.set_ylabel("Proporción")
        ax.set_title("Distribución de signos en las 30 quinielas")
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
    else:
        st.error("No se pudo cargar los datos para la distribución")

def mostrar_metricas(sim):
    """Muestra las métricas probabilísticas"""
    if sim is not None:
        st.subheader("📈 Métricas Probabilísticas del Portafolio")
        
        # Verificar que las columnas existen
        if all(col in sim.columns for col in ['pr_11', 'pr_10', 'mu', 'sigma']):
            # Separar columnas numéricas de las de texto
            columnas_numericas = ['mu', 'sigma', 'pr_10', 'pr_11']
            columnas_texto = [col for col in sim.columns if col not in columnas_numericas]
            
            # Crear una copia del DataFrame para mostrar
            sim_display = sim.copy()
            
            # Convertir columnas numéricas a float para evitar errores de formato
            for col in columnas_numericas:
                if col in sim_display.columns:
                    sim_display[col] = pd.to_numeric(sim_display[col], errors='coerce')
            
            # Aplicar formato solo a columnas numéricas
            styled_df = sim_display.style.format({
                col: "{:.3f}" for col in columnas_numericas if col in sim_display.columns
            })
            
            st.dataframe(styled_df, use_container_width=True)

            # Métricas principales - usando valores convertidos a float
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                pr11_mean = pd.to_numeric(sim["pr_11"], errors='coerce').mean()
                st.metric(label="🎯 Pr[≥11] esperada", value=f"{pr11_mean:.2%}")
            
            with col2:
                pr10_mean = pd.to_numeric(sim["pr_10"], errors='coerce').mean()
                st.metric(label="🎯 Pr[≥10] esperada", value=f"{pr10_mean:.2%}")
            
            with col3:
                mu_mean = pd.to_numeric(sim["mu"], errors='coerce').mean()
                st.metric(label="🔢 μ hits esperados", value=f"{mu_mean:.2f}")
                
            with col4:
                sigma_mean = pd.to_numeric(sim["sigma"], errors='coerce').mean()
                st.metric(label="📊 σ varianza", value=f"{sigma_mean:.2f}")

            # ROI estimado
            st.subheader("💰 Análisis Financiero")
            costo_total = 30 * 15  # 30 boletos x $15
            ganancia_esperada = pr11_mean * 90000  # Premio estimado
            roi = (ganancia_esperada / costo_total - 1) * 100
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("💵 Inversión Total", f"${costo_total:,}")
            with col2:
                st.metric("🎁 Ganancia Esperada", f"${ganancia_esperada:,.0f}")
            with col3:
                st.metric("📈 ROI Estimado", f"{roi:.1f}%")
                
        else:
            st.error("❌ El archivo de simulación no tiene las columnas correctas")
            st.info("Columnas encontradas: " + ", ".join(sim.columns))
    else:
        st.error("No se pudo cargar las métricas de simulación")

def descargar_portafolio(port):
    """Permite descargar el portafolio"""
    if port is not None:
        csv = port.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Descargar CSV", csv, "quinielas_2283.csv", "text/csv")

def main():
    """Función principal del dashboard"""
    st.title("🔢 Dashboard de Portafolio Progol")
    st.markdown("---")
    
    # Verificar y corregir datos antes de cargar
    with st.expander("🔧 Verificación y Corrección de Datos", expanded=True):
        if st.button("🔍 Verificar Archivos"):
            with st.spinner("Verificando archivos..."):
                datos_ok = verificar_y_corregir_datos()
                
                if datos_ok:
                    st.success("✅ Todos los datos están listos")
                    st.rerun()  # Reiniciar la app para cargar los datos corregidos
    
    st.markdown("---")
    
    # Cargar y mostrar datos
    port, prob, sim = cargar_datos_seguros()
    
    if port is not None and sim is not None:
        # Mostrar componentes del dashboard
        mostrar_quinielas(port)
        st.markdown("---")
        
        mostrar_distribucion_signos(port)
        st.markdown("---")
        
        mostrar_metricas(sim)
        st.markdown("---")
        
        descargar_portafolio(port)
        
        # Información adicional
        with st.expander("ℹ️ Información Técnica"):
            st.info(f"""
            **Archivos cargados:**
            - Portafolio: {len(port)} quinielas
            - Simulación: {len(sim)} métricas
            - Columnas disponibles: {', '.join(sim.columns)}
            """)
    else:
        st.error("❌ No se pudieron cargar los datos necesarios")
        st.info("Usa el botón 'Verificar Archivos' arriba para diagnosticar el problema")

if __name__ == "__main__":
    main()
