#!/usr/bin/env python3
"""
Simulador Monte Carlo mejorado - Progol Engine v2
Cálculo preciso de métricas de portafolio para 21 partidos
"""

import numpy as np
import pandas as pd
import json
import logging
from pathlib import Path
import time
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from datetime import datetime
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SimulationConfig:
    """Configuración de simulación"""
    n_simulations: int = 50000
    n_workers: int = None  # None = auto-detect
    batch_size: int = 5000
    random_seed: int = 42
    confidence_level: float = 0.95
    include_correlations: bool = False
    precision_mode: str = 'standard'  # 'fast', 'standard', 'high'

@dataclass
class QuinielaMetrics:
    """Métricas de una quiniela"""
    quiniela_id: str
    mu: float                    # Media de aciertos
    sigma: float                 # Desviación estándar
    pr_8: float                  # Pr[≥8 aciertos]
    pr_9: float                  # Pr[≥9 aciertos]
    pr_10: float                 # Pr[≥10 aciertos]
    pr_11: float                 # Pr[≥11 aciertos]
    pr_12: float                 # Pr[≥12 aciertos]
    pr_13: float                 # Pr[≥13 aciertos]
    pr_14: float                 # Pr[≥14 aciertos]
    percentile_5: float          # Percentil 5%
    percentile_95: float         # Percentil 95%
    mode: int                    # Moda (aciertos más probable)
    skewness: float              # Asimetría
    kurtosis: float              # Curtosis
    
    # Métricas financieras
    expected_return: float       # Retorno esperado
    value_at_risk_95: float     # VaR 95%
    sharpe_ratio: float         # Ratio Sharpe (ajustado)

@dataclass
class PortfolioMetrics:
    """Métricas del portafolio completo"""
    portfolio_id: str
    n_quinielas: int
    
    # Métricas principales
    portfolio_pr_11: float       # Pr[≥1 premio] del portafolio
    expected_prizes: float       # Número esperado de premios
    prize_probability: float     # Probabilidad de al menos 1 premio
    
    # Distribución de premios
    pr_0_premios: float         # Pr[0 premios]
    pr_1_premio: float          # Pr[1 premio]
    pr_2_premios: float         # Pr[2 premios]
    pr_3_plus_premios: float    # Pr[≥3 premios]
    
    # Métricas financieras
    total_cost: float           # Costo total
    expected_revenue: float     # Ingresos esperados
    expected_profit: float      # Ganancia esperada
    roi_expected: float         # ROI esperado
    roi_95_ci: Tuple[float, float]  # Intervalo confianza ROI
    break_even_prob: float      # Probabilidad break-even
    
    # Métricas de riesgo
    portfolio_var_95: float     # VaR del portafolio
    max_loss_prob: float       # Probabilidad pérdida máxima
    profit_probability: float   # Probabilidad ganancia
    
    # Correlaciones
    avg_correlation: float      # Correlación promedio entre quinielas
    diversification_ratio: float # Ratio diversificación

class MonteCarloSimulator:
    """Simulador Monte Carlo mejorado"""
    
    def __init__(self, config: SimulationConfig = None):
        self.config = config or SimulationConfig()
        
        # Configurar número de workers
        if self.config.n_workers is None:
            self.config.n_workers = min(8, mp.cpu_count())
        
        # Configurar precisión
        if self.config.precision_mode == 'fast':
            self.config.n_simulations = min(10000, self.config.n_simulations)
            self.config.batch_size = 2000
        elif self.config.precision_mode == 'high':
            self.config.n_simulations = max(100000, self.config.n_simulations)
            self.config.batch_size = 10000
        
        # Configurar semilla
        np.random.seed(self.config.random_seed)
        
        logger.info(f"Simulador configurado: {self.config.n_simulations:,} sims, "
                   f"{self.config.n_workers} workers, batch={self.config.batch_size:,}")
    
    def simulate_quiniela(self, quiniela_probs: List[float], 
                         quiniela_id: str = "Q01") -> QuinielaMetrics:
        """Simular una quiniela individual"""
        n_partidos = len(quiniela_probs)
        n_sims = self.config.n_simulations
        
        # Generar todas las simulaciones
        random_matrix = np.random.random((n_sims, n_partidos))
        prob_matrix = np.array(quiniela_probs)
        
        # Calcular aciertos por simulación
        hits_matrix = (random_matrix < prob_matrix).astype(int)
        hits_per_sim = hits_matrix.sum(axis=1)
        
        # Calcular métricas básicas
        mu = float(np.mean(hits_per_sim))
        sigma = float(np.std(hits_per_sim))
        
        # Probabilidades de diferentes niveles de aciertos
        pr_metrics = {}
        for threshold in [8, 9, 10, 11, 12, 13, 14]:
            if threshold <= n_partidos:
                pr_metrics[f'pr_{threshold}'] = float(np.mean(hits_per_sim >= threshold))
            else:
                pr_metrics[f'pr_{threshold}'] = 0.0
        
        # Percentiles
        percentile_5 = float(np.percentile(hits_per_sim, 5))
        percentile_95 = float(np.percentile(hits_per_sim, 95))
        
        # Moda
        unique, counts = np.unique(hits_per_sim, return_counts=True)
        mode = int(unique[np.argmax(counts)])
        
        # Momentos estadísticos
        skewness = float(stats.skew(hits_per_sim))
        kurtosis = float(stats.kurtosis(hits_per_sim))
        
        # Métricas financieras
        costo_boleto = 15
        premio_cat2 = 90000
        
        # Retornos por simulación
        returns = np.where(hits_per_sim >= 11, premio_cat2 - costo_boleto, -costo_boleto)
        expected_return = float(np.mean(returns))
        value_at_risk_95 = float(np.percentile(returns, 5))  # VaR 95%
        
        # Sharpe ratio (ajustado)
        return_std = np.std(returns)
        sharpe_ratio = expected_return / return_std if return_std > 0 else 0
        
        return QuinielaMetrics(
            quiniela_id=quiniela_id,
            mu=mu,
            sigma=sigma,
            pr_8=pr_metrics['pr_8'],
            pr_9=pr_metrics['pr_9'],
            pr_10=pr_metrics['pr_10'],
            pr_11=pr_metrics['pr_11'],
            pr_12=pr_metrics['pr_12'],
            pr_13=pr_metrics['pr_13'],
            pr_14=pr_metrics['pr_14'],
            percentile_5=percentile_5,
            percentile_95=percentile_95,
            mode=mode,
            skewness=skewness,
            kurtosis=kurtosis,
            expected_return=expected_return,
            value_at_risk_95=value_at_risk_95,
            sharpe_ratio=sharpe_ratio
        )
    
    def simulate_portfolio(self, portfolio_probs: List[List[float]], 
                          quiniela_ids: List[str] = None,
                          portfolio_id: str = "Portfolio_01") -> Tuple[List[QuinielaMetrics], PortfolioMetrics]:
        """Simular portafolio completo"""
        logger.info(f"Iniciando simulación de portafolio: {len(portfolio_probs)} quinielas")
        
        start_time = time.time()
        
        # IDs por defecto
        if quiniela_ids is None:
            quiniela_ids = [f"Q{i+1:02d}" for i in range(len(portfolio_probs))]
        
        # Simular cada quiniela individualmente
        quiniela_metrics = []
        
        if self.config.n_workers > 1:
            # Simulación paralela
            quiniela_metrics = self._simulate_parallel(portfolio_probs, quiniela_ids)
        else:
            # Simulación secuencial
            for i, probs in enumerate(portfolio_probs):
                metrics = self.simulate_quiniela(probs, quiniela_ids[i])
                quiniela_metrics.append(metrics)
        
        # Simular interacciones del portafolio
        portfolio_metrics = self._simulate_portfolio_interactions(
            portfolio_probs, quiniela_metrics, portfolio_id
        )
        
        elapsed_time = time.time() - start_time
        logger.info(f"Simulación completada en {elapsed_time:.1f}s")
        
        return quiniela_metrics, portfolio_metrics
    
    def _simulate_parallel(self, portfolio_probs: List[List[float]], 
                          quiniela_ids: List[str]) -> List[QuinielaMetrics]:
        """Simulación paralela de quinielas"""
        logger.info(f"Simulación paralela con {self.config.n_workers} workers")
        
        with ProcessPoolExecutor(max_workers=self.config.n_workers) as executor:
            # Enviar trabajos
            futures = []
            for i, probs in enumerate(portfolio_probs):
                future = executor.submit(self._simulate_quiniela_worker, probs, quiniela_ids[i])
                futures.append(future)
            
            # Recoger resultados
            results = []
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error en simulación paralela: {e}")
                    # Fallback a simulación secuencial para esta quiniela
                    idx = futures.index(future)
                    result = self.simulate_quiniela(portfolio_probs[idx], quiniela_ids[idx])
                    results.append(result)
        
        # Ordenar por quiniela_id
        results.sort(key=lambda x: x.quiniela_id)
        return results
    
    def _simulate_quiniela_worker(self, probs: List[float], quiniela_id: str) -> QuinielaMetrics:
        """Worker para simulación paralela"""
        # Crear nuevo simulador con la misma configuración
        worker_config = SimulationConfig(
            n_simulations=self.config.n_simulations,
            n_workers=1,  # Worker individual
            batch_size=self.config.batch_size,
            random_seed=None,  # Permitir semilla aleatoria
            confidence_level=self.config.confidence_level,
            include_correlations=False,
            precision_mode=self.config.precision_mode
        )
        
        worker_sim = MonteCarloSimulator(worker_config)
        return worker_sim.simulate_quiniela(probs, quiniela_id)
    
    def _simulate_portfolio_interactions(self, portfolio_probs: List[List[float]], 
                                       quiniela_metrics: List[QuinielaMetrics],
                                       portfolio_id: str) -> PortfolioMetrics:
        """Simular interacciones y métricas del portafolio"""
        n_quinielas = len(portfolio_probs)
        n_partidos = len(portfolio_probs[0]) if portfolio_probs else 0
        n_sims = self.config.n_simulations
        
        # Configurar costos y premios
        costo_boleto = 15
        premio_cat2 = 90000
        total_cost = n_quinielas * costo_boleto
        
        # Matriz de probabilidades del portafolio
        portfolio_matrix = np.array(portfolio_probs)  # (n_quinielas, n_partidos)
        
        # Generar matriz de resultados aleatorios
        # (n_sims, n_partidos) - cada fila es una realización de todos los partidos
        match_results = np.random.random((n_sims, n_partidos))
        
        # Para cada simulación, calcular ganadores en cada quiniela
        premios_por_sim = np.zeros(n_sims)
        hits_matrix = np.zeros((n_sims, n_quinielas))
        
        for sim_idx in range(n_sims):
            # Resultados de partidos en esta simulación
            match_outcomes = match_results[sim_idx]
            
            # Calcular aciertos en cada quiniela
            for q_idx in range(n_quinielas):
                quiniela_probs = portfolio_matrix[q_idx]
                hits = np.sum(match_outcomes < quiniela_probs)
                hits_matrix[sim_idx, q_idx] = hits
                
                # Verificar si ganó premio (≥11 aciertos)
                if hits >= 11:
                    premios_por_sim[sim_idx] += 1
        
        # Calcular métricas del portafolio
        
        # 1. Distribución de premios
        pr_0_premios = float(np.mean(premios_por_sim == 0))
        pr_1_premio = float(np.mean(premios_por_sim == 1))
        pr_2_premios = float(np.mean(premios_por_sim == 2))
        pr_3_plus_premios = float(np.mean(premios_por_sim >= 3))
        
        # 2. Métricas principales
        expected_prizes = float(np.mean(premios_por_sim))
        prize_probability = float(np.mean(premios_por_sim >= 1))
        portfolio_pr_11 = prize_probability  # Pr[≥1 premio]
        
        # 3. Métricas financieras
        revenues_per_sim = premios_por_sim * premio_cat2
        profits_per_sim = revenues_per_sim - total_cost
        
        expected_revenue = float(np.mean(revenues_per_sim))
        expected_profit = float(np.mean(profits_per_sim))
        roi_expected = (expected_profit / total_cost) * 100
        
        # Intervalo de confianza para ROI
        roi_per_sim = (profits_per_sim / total_cost) * 100
        roi_ci_lower = float(np.percentile(roi_per_sim, (1 - self.config.confidence_level) / 2 * 100))
        roi_ci_upper = float(np.percentile(roi_per_sim, (1 + self.config.confidence_level) / 2 * 100))
        roi_95_ci = (roi_ci_lower, roi_ci_upper)
        
        # 4. Métricas de riesgo
        portfolio_var_95 = float(np.percentile(profits_per_sim, 5))  # VaR 95%
        max_loss_prob = float(np.mean(profits_per_sim == -total_cost))  # Prob pérdida total
        profit_probability = float(np.mean(profits_per_sim > 0))  # Prob ganancia
        break_even_prob = float(np.mean(profits_per_sim >= 0))  # Prob break-even
        
        # 5. Correlaciones entre quinielas
        if n_quinielas > 1:
            corr_matrix = np.corrcoef(hits_matrix.T)
            # Promedio de correlaciones (excluyendo diagonal)
            mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
            avg_correlation = float(np.mean(corr_matrix[mask]))
            
            # Ratio de diversificación (aproximado)
            portfolio_variance = np.var(np.sum(hits_matrix, axis=1))
            individual_variances = np.sum([np.var(hits_matrix[:, i]) for i in range(n_quinielas)])
            diversification_ratio = float(portfolio_variance / individual_variances) if individual_variances > 0 else 1.0
        else:
            avg_correlation = 0.0
            diversification_ratio = 1.0
        
        return PortfolioMetrics(
            portfolio_id=portfolio_id,
            n_quinielas=n_quinielas,
            portfolio_pr_11=portfolio_pr_11,
            expected_prizes=expected_prizes,
            prize_probability=prize_probability,
            pr_0_premios=pr_0_premios,
            pr_1_premio=pr_1_premio,
            pr_2_premios=pr_2_premios,
            pr_3_plus_premios=pr_3_plus_premios,
            total_cost=total_cost,
            expected_revenue=expected_revenue,
            expected_profit=expected_profit,
            roi_expected=roi_expected,
            roi_95_ci=roi_95_ci,
            break_even_prob=break_even_prob,
            portfolio_var_95=portfolio_var_95,
            max_loss_prob=max_loss_prob,
            profit_probability=profit_probability,
            avg_correlation=avg_correlation,
            diversification_ratio=diversification_ratio
        )
    
    def sensitivity_analysis(self, base_probs: List[List[float]], 
                           sensitivity_range: float = 0.1) -> Dict[str, Any]:
        """Análisis de sensibilidad del portafolio"""
        logger.info("Ejecutando análisis de sensibilidad...")
        
        results = {
            'base_scenario': None,
            'optimistic_scenario': None,
            'pessimistic_scenario': None,
            'sensitivity_metrics': {}
        }
        
        # Escenario base
        _, base_portfolio = self.simulate_portfolio(base_probs, portfolio_id="Base")
        results['base_scenario'] = base_portfolio
        
        # Escenario optimista (+10% en todas las probabilidades)
        optimistic_probs = []
        for quiniela_probs in base_probs:
            opt_probs = [min(1.0, p * (1 + sensitivity_range)) for p in quiniela_probs]
            optimistic_probs.append(opt_probs)
        
        _, opt_portfolio = self.simulate_portfolio(optimistic_probs, portfolio_id="Optimistic")
        results['optimistic_scenario'] = opt_portfolio
        
        # Escenario pesimista (-10% en todas las probabilidades)
        pessimistic_probs = []
        for quiniela_probs in base_probs:
            pess_probs = [max(0.0, p * (1 - sensitivity_range)) for p in quiniela_probs]
            pessimistic_probs.append(pess_probs)
        
        _, pess_portfolio = self.simulate_portfolio(pessimistic_probs, portfolio_id="Pessimistic")
        results['pessimistic_scenario'] = pess_portfolio
        
        # Métricas de sensibilidad
        results['sensitivity_metrics'] = {
            'roi_range': opt_portfolio.roi_expected - pess_portfolio.roi_expected,
            'pr11_range': opt_portfolio.portfolio_pr_11 - pess_portfolio.portfolio_pr_11,
            'profit_range': opt_portfolio.expected_profit - pess_portfolio.expected_profit,
            'risk_increase': pess_portfolio.max_loss_prob - base_portfolio.max_loss_prob,
            'upside_potential': opt_portfolio.profit_probability - base_portfolio.profit_probability
        }
        
        logger.info("Análisis de sensibilidad completado")
        return results

class SimulationExporter:
    """Exportador de resultados de simulación"""
    
    @staticmethod
    def export_quiniela_metrics(metrics: List[QuinielaMetrics], 
                               output_path: str) -> str:
        """Exportar métricas de quinielas a CSV"""
        data = []
        for metric in metrics:
            data.append({
                'quiniela_id': metric.quiniela_id,
                'mu': metric.mu,
                'sigma': metric.sigma,
                'pr_8': metric.pr_8,
                'pr_9': metric.pr_9,
                'pr_10': metric.pr_10,
                'pr_11': metric.pr_11,
                'pr_12': metric.pr_12,
                'pr_13': metric.pr_13,
                'pr_14': metric.pr_14,
                'percentile_5': metric.percentile_5,
                'percentile_95': metric.percentile_95,
                'mode': metric.mode,
                'skewness': metric.skewness,
                'kurtosis': metric.kurtosis,
                'expected_return': metric.expected_return,
                'value_at_risk_95': metric.value_at_risk_95,
                'sharpe_ratio': metric.sharpe_ratio
            })
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        logger.info(f"Métricas de quinielas exportadas: {output_path}")
        return output_path
    
    @staticmethod
    def export_portfolio_metrics(portfolio_metrics: PortfolioMetrics, 
                                output_path: str) -> str:
        """Exportar métricas de portafolio a JSON"""
        data = {
            'portfolio_id': portfolio_metrics.portfolio_id,
            'n_quinielas': portfolio_metrics.n_quinielas,
            'timestamp': datetime.now().isoformat(),
            
            # Métricas principales
            'portfolio_pr_11': portfolio_metrics.portfolio_pr_11,
            'expected_prizes': portfolio_metrics.expected_prizes,
            'prize_probability': portfolio_metrics.prize_probability,
            
            # Distribución de premios
            'prize_distribution': {
                'pr_0_premios': portfolio_metrics.pr_0_premios,
                'pr_1_premio': portfolio_metrics.pr_1_premio,
                'pr_2_premios': portfolio_metrics.pr_2_premios,
                'pr_3_plus_premios': portfolio_metrics.pr_3_plus_premios
            },
            
            # Métricas financieras
            'financial_metrics': {
                'total_cost': portfolio_metrics.total_cost,
                'expected_revenue': portfolio_metrics.expected_revenue,
                'expected_profit': portfolio_metrics.expected_profit,
                'roi_expected': portfolio_metrics.roi_expected,
                'roi_95_ci_lower': portfolio_metrics.roi_95_ci[0],
                'roi_95_ci_upper': portfolio_metrics.roi_95_ci[1],
                'break_even_prob': portfolio_metrics.break_even_prob
            },
            
            # Métricas de riesgo
            'risk_metrics': {
                'portfolio_var_95': portfolio_metrics.portfolio_var_95,
                'max_loss_prob': portfolio_metrics.max_loss_prob,
                'profit_probability': portfolio_metrics.profit_probability
            },
            
            # Correlaciones
            'correlation_metrics': {
                'avg_correlation': portfolio_metrics.avg_correlation,
                'diversification_ratio': portfolio_metrics.diversification_ratio
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Métricas de portafolio exportadas: {output_path}")
        return output_path
    
    @staticmethod
    def export_sensitivity_analysis(sensitivity_results: Dict[str, Any], 
                                   output_path: str) -> str:
        """Exportar análisis de sensibilidad a JSON"""
        # Convertir métricas de portafolio a diccionarios
        def portfolio_to_dict(portfolio_metrics: PortfolioMetrics) -> Dict:
            return {
                'portfolio_pr_11': portfolio_metrics.portfolio_pr_11,
                'roi_expected': portfolio_metrics.roi_expected,
                'expected_profit': portfolio_metrics.expected_profit,
                'profit_probability': portfolio_metrics.profit_probability,
                'max_loss_prob': portfolio_metrics.max_loss_prob
            }
        
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'scenarios': {
                'base': portfolio_to_dict(sensitivity_results['base_scenario']),
                'optimistic': portfolio_to_dict(sensitivity_results['optimistic_scenario']),
                'pessimistic': portfolio_to_dict(sensitivity_results['pessimistic_scenario'])
            },
            'sensitivity_metrics': sensitivity_results['sensitivity_metrics']
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Análisis de sensibilidad exportado: {output_path}")
        return output_path

def load_portfolio_from_files(portfolio_file: str, prob_file: str) -> Tuple[List[List[float]], List[str]]:
    """Cargar portafolio desde archivos CSV"""
    # Cargar portafolio
    df_portfolio = pd.read_csv(portfolio_file)
    
    # Cargar probabilidades
    df_probs = pd.read_csv(prob_file)
    
    # Mapear probabilidades por partido
    prob_map = {}
    for _, row in df_probs.iterrows():
        partido = row['partido']
        prob_map[partido] = {
            'L': row['p_final_L'],
            'E': row['p_final_E'],
            'V': row['p_final_V']
        }
    
    # Construir matriz de probabilidades del portafolio
    portfolio_probs = []
    quiniela_ids = []
    
    # Detectar columnas de partidos
    partido_cols = [col for col in df_portfolio.columns if col.startswith('P') and col[1:].isdigit()]
    partido_cols.sort(key=lambda x: int(x[1:]))
    
    for _, row in df_portfolio.iterrows():
        quiniela_ids.append(row['quiniela_id'])
        
        quiniela_probs = []
        for col in partido_cols:
            partido_num = int(col[1:])
            signo = row[col]
            
            if partido_num in prob_map:
                prob = prob_map[partido_num][signo]
                quiniela_probs.append(prob)
            else:
                # Probabilidad por defecto si no hay datos
                default_probs = {'L': 0.4, 'E': 0.3, 'V': 0.3}
                quiniela_probs.append(default_probs[signo])
        
        portfolio_probs.append(quiniela_probs)
    
    logger.info(f"Portafolio cargado: {len(portfolio_probs)} quinielas, "
               f"{len(portfolio_probs[0]) if portfolio_probs else 0} partidos")
    
    return portfolio_probs, quiniela_ids

def run_complete_simulation(jornada: Union[str, int], 
                           precision_mode: str = 'standard',
                           include_sensitivity: bool = True) -> Dict[str, str]:
    """Ejecutar simulación completa desde archivos"""
    logger.info(f"=== INICIANDO SIMULACIÓN COMPLETA - JORNADA {jornada} ===")
    
    start_time = time.time()
    output_paths = {}
    
    try:
        # Rutas de archivos
        portfolio_file = f"data/processed/portfolio_final_{jornada}.csv"
        prob_file = f"data/processed/prob_final_{jornada}.csv"
        
        # Verificar archivos
        if not Path(portfolio_file).exists():
            raise FileNotFoundError(f"Portafolio no encontrado: {portfolio_file}")
        
        if not Path(prob_file).exists():
            raise FileNotFoundError(f"Probabilidades no encontradas: {prob_file}")
        
        # Cargar datos
        portfolio_probs, quiniela_ids = load_portfolio_from_files(portfolio_file, prob_file)
        
        # Configurar simulador
        config = SimulationConfig(
            precision_mode=precision_mode,
            n_simulations=50000 if precision_mode == 'standard' else 100000
        )
        
        simulator = MonteCarloSimulator(config)
        
        # Ejecutar simulación principal
        quiniela_metrics, portfolio_metrics = simulator.simulate_portfolio(
            portfolio_probs, quiniela_ids, f"Portfolio_{jornada}"
        )
        
        # Exportar resultados
        exporter = SimulationExporter()
        
        # 1. Métricas de quinielas
        quiniela_path = f"data/processed/simulation_metrics_{jornada}.csv"
        exporter.export_quiniela_metrics(quiniela_metrics, quiniela_path)
        output_paths['quiniela_metrics'] = quiniela_path
        
        # 2. Métricas de portafolio
        portfolio_path = f"data/processed/portfolio_metrics_{jornada}.json"
        exporter.export_portfolio_metrics(portfolio_metrics, portfolio_path)
        output_paths['portfolio_metrics'] = portfolio_path
        
        # 3. Análisis de sensibilidad (opcional)
        if include_sensitivity:
            logger.info("Ejecutando análisis de sensibilidad...")
            sensitivity_results = simulator.sensitivity_analysis(portfolio_probs)
            
            sensitivity_path = f"data/processed/sensitivity_analysis_{jornada}.json"
            exporter.export_sensitivity_analysis(sensitivity_results, sensitivity_path)
            output_paths['sensitivity_analysis'] = sensitivity_path
        
        elapsed_time = time.time() - start_time
        
        # Log de resumen
        logger.info("=== SIMULACIÓN COMPLETADA ===")
        logger.info(f"Tiempo total: {elapsed_time:.1f}s")
        logger.info(f"Pr[≥11] Portafolio: {portfolio_metrics.portfolio_pr_11:.4f}")
        logger.info(f"ROI Esperado: {portfolio_metrics.roi_expected:.1f}%")
        logger.info(f"Ganancia Esperada: ${portfolio_metrics.expected_profit:,.0f}")
        
        return output_paths
        
    except Exception as e:
        logger.error(f"❌ Error en simulación: {e}")
        raise

if __name__ == "__main__":
    import sys
    
    # Parámetros desde línea de comandos
    jornada = sys.argv[1] if len(sys.argv) > 1 else "2283"
    precision = sys.argv[2] if len(sys.argv) > 2 else "standard"
    sensitivity = sys.argv[3].lower() == "true" if len(sys.argv) > 3 else True
    
    # Ejecutar simulación
    try:
        output_paths = run_complete_simulation(jornada, precision, sensitivity)
        
        print("\n🎉 SIMULACIÓN MONTE CARLO COMPLETADA!")
        print("📁 Archivos generados:")
        for file_type, path in output_paths.items():
            print(f"  • {file_type}: {path}")
            
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        sys.exit(1)