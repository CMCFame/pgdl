#!/usr/bin/env python3
"""
Configuración actualizada para Progol Engine v2
Soporte para 21 partidos (14 regulares + 7 revancha)
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import json
from typing import Dict, List, Optional, Union

# Cargar variables de entorno
load_dotenv()

# ===== CONFIGURACIÓN BÁSICA =====

# Información del proyecto
PROJECT_NAME = "Progol Engine v2"
VERSION = "2.0.0"
DESCRIPTION = "Sistema Avanzado de Optimización de Quinielas con Soporte para Revancha"

# Directorios base
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# ===== CONFIGURACIÓN DE PARTIDOS =====

class PartidosConfig:
    """Configuración para manejo de partidos regulares y revancha"""
    
    # Partidos fijos
    PARTIDOS_REGULARES = 14
    PARTIDOS_REVANCHA_MAX = 7
    PARTIDOS_TOTAL_MAX = PARTIDOS_REGULARES + PARTIDOS_REVANCHA_MAX  # 21
    
    # Configuración por defecto
    DEFAULT_CONFIG = {
        'regulares': PARTIDOS_REGULARES,
        'revancha': 0,
        'total': PARTIDOS_REGULARES
    }
    
    @classmethod
    def validate_config(cls, config: Dict) -> bool:
        """Validar configuración de partidos"""
        regulares = config.get('regulares', 0)
        revancha = config.get('revancha', 0)
        total = config.get('total', 0)
        
        # Validaciones
        if regulares != cls.PARTIDOS_REGULARES:
            return False
        
        if revancha < 0 or revancha > cls.PARTIDOS_REVANCHA_MAX:
            return False
        
        if total != regulares + revancha:
            return False
        
        return True
    
    @classmethod
    def get_column_names(cls, n_partidos: int) -> List[str]:
        """Generar nombres de columnas para partidos"""
        return [f'P{i+1}' for i in range(n_partidos)]
    
    @classmethod
    def get_match_types(cls, n_partidos: int) -> List[str]:
        """Generar tipos de partido (Regular/Revancha)"""
        tipos = []
        for i in range(n_partidos):
            if i < cls.PARTIDOS_REGULARES:
                tipos.append('Regular')
            else:
                tipos.append('Revancha')
        return tipos

# ===== CONFIGURACIÓN DE ARCHIVOS =====

class FileConfig:
    """Configuración de archivos y rutas"""
    
    # Directorios principales
    RAW_DATA_PATH = os.getenv("DATA_RAW_PATH", "data/raw/")
    PROCESSED_DATA_PATH = os.getenv("DATA_PROCESSED_PATH", "data/processed/")
    DASHBOARD_DATA_PATH = os.getenv("DASHBOARD_DATA_PATH", "data/dashboard/")
    OUTPUTS_PATH = os.getenv("OUTPUTS_PATH", "data/outputs/")
    JSON_PREVIAS_PATH = os.getenv("JSON_PREVIAS_PATH", "data/json_previas/")
    REPORTS_PATH = os.getenv("REPORTS_PATH", "data/reports/")
    
    # Archivos base por jornada
    PROGOL_CSV = os.getenv("PROGOL_CSV", "Progol.csv")
    ODDS_CSV = os.getenv("ODDS_CSV", "odds.csv")
    ELO_CSV = os.getenv("ELO_CSV", "elo.csv")
    SQUAD_CSV = os.getenv("SQUAD_CSV", "squad_value.csv")
    PREVIAS_PDF = os.getenv("PREVIAS_PDF", "previas.pdf")
    
    # Plantillas de archivos de salida
    OUTPUT_TEMPLATES = {
        'matches': "{data_path}/matches_{jornada}.csv",
        'probabilities': "{data_path}/prob_final_{jornada}.csv",
        'portfolio': "{data_path}/portfolio_final_{jornada}.csv",
        'simulation': "{data_path}/simulation_metrics_{jornada}.csv",
        'features': "{data_path}/match_features_{jornada}.feather",
        'model_metrics': "{data_path}/model_metrics_{jornada}.json",
        'portfolio_stats': "{data_path}/portfolio_stats_{jornada}.json"
    }
    
    @classmethod
    def get_output_path(cls, file_type: str, jornada: Union[str, int], 
                       data_path: str = None) -> str:
        """Generar ruta de archivo de salida"""
        if data_path is None:
            data_path = cls.PROCESSED_DATA_PATH
        
        template = cls.OUTPUT_TEMPLATES.get(file_type)
        if not template:
            raise ValueError(f"Tipo de archivo no reconocido: {file_type}")
        
        return template.format(data_path=data_path, jornada=jornada)
    
    @classmethod
    def ensure_directories(cls):
        """Crear directorios necesarios"""
        directories = [
            cls.RAW_DATA_PATH,
            cls.PROCESSED_DATA_PATH,
            cls.DASHBOARD_DATA_PATH,
            cls.OUTPUTS_PATH,
            cls.JSON_PREVIAS_PATH,
            cls.REPORTS_PATH
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

# ===== CONFIGURACIÓN DEL PIPELINE =====

class PipelineConfig:
    """Configuración del pipeline de procesamiento"""
    
    # Parámetros generales
    N_QUINIELAS = int(os.getenv("N_QUINIELAS", "30"))
    COSTO_BOLETO = float(os.getenv("COSTO_BOLETO", "15"))
    PREMIO_CAT2 = float(os.getenv("PREMIO_CAT2", "90000"))
    N_MONTECARLO_SAMPLES = int(os.getenv("N_MONTECARLO_SAMPLES", "50000"))
    PR11_UMBRAL = float(os.getenv("PR11_UMBRAL", "0.10"))
    
    # Distribución histórica de signos (Progol)
    DISTRIBUCION_HISTORICA = {
        'L': 0.38,  # Local
        'E': 0.29,  # Empate  
        'V': 0.33   # Visitante
    }
    
    # Límites de concentración
    MAX_CONCENTRACION_GENERAL = 0.70  # Máx 70% del mismo signo por boleto
    MAX_CONCENTRACION_PRIMEROS_3 = 0.60  # Máx 60% en primeros 3 partidos
    
    # Empates obligatorios por boleto
    MIN_EMPATES_POR_BOLETO = 4
    MAX_EMPATES_POR_BOLETO = 6
    
    # Configuración arquitectura Core + Satélites
    N_CORE_QUINIELAS = 4
    N_SATELITES = N_QUINIELAS - N_CORE_QUINIELAS
    
    @classmethod
    def validate_portfolio_rules(cls, df_portfolio, n_partidos: int) -> Dict[str, bool]:
        """Validar reglas del portafolio"""
        validations = {}
        
        # Verificar número de quinielas
        validations['n_quinielas'] = len(df_portfolio) == cls.N_QUINIELAS
        
        # Verificar distribución de signos por boleto
        concentracion_ok = True
        empates_ok = True
        
        for _, row in df_portfolio.iterrows():
            # Contar signos
            signos = [row[f'P{i+1}'] for i in range(n_partidos)]
            counts = {sign: signos.count(sign) for sign in ['L', 'E', 'V']}
            
            # Verificar concentración
            max_count = max(counts.values())
            if max_count / len(signos) > cls.MAX_CONCENTRACION_GENERAL:
                concentracion_ok = False
            
            # Verificar empates
            if not (cls.MIN_EMPATES_POR_BOLETO <= counts['E'] <= cls.MAX_EMPATES_POR_BOLETO):
                empates_ok = False
        
        validations['concentracion'] = concentracion_ok
        validations['empates'] = empates_ok
        
        return validations

# ===== CONFIGURACIÓN BAYESIANA =====

class BayesianConfig:
    """Configuración de parámetros bayesianos"""
    
    # Coeficientes por resultado (k1: forma, k2: lesiones, k3: contexto)
    COEFICIENTES_L = {
        'k1': float(os.getenv("K1_L", "0.15")),   # Forma local
        'k2': float(os.getenv("K2_L", "-0.12")),  # Lesiones local
        'k3': float(os.getenv("K3_L", "0.08"))    # Contexto local
    }
    
    COEFICIENTES_E = {
        'k1': float(os.getenv("K1_E", "-0.10")),  # Forma empate
        'k2': float(os.getenv("K2_E", "0.15")),   # Lesiones empate
        'k3': float(os.getenv("K3_E", "0.03"))    # Contexto empate
    }
    
    COEFICIENTES_V = {
        'k1': float(os.getenv("K1_V", "-0.08")),  # Forma visitante
        'k2': float(os.getenv("K2_V", "-0.10")),  # Lesiones visitante
        'k3': float(os.getenv("K3_V", "-0.05"))   # Contexto visitante
    }
    
    # Configuración del modelo Poisson
    POISSON_CONFIG = {
        'lambda_1_prior': 1.4,  # Goles local
        'lambda_2_prior': 1.1,  # Goles visitante
        'lambda_3_prior': 0.3,  # Covarianza
        'elo_factor': 0.003     # Factor ELO
    }
    
    # Pesos de stacking
    STACKING_WEIGHTS = {
        'odds': 0.58,      # Peso probabilidades implícitas
        'poisson': 0.42    # Peso modelo Poisson
    }
    
    # Regla de empates
    DRAW_RULE = {
        'threshold': 0.08,    # Umbral diferencia L-V
        'boost_empate': 0.06, # Incremento probabilidad empate
        'reduce_lv': 0.03     # Reducción L y V
    }

# ===== CONFIGURACIÓN DE OPTIMIZACIÓN =====

class OptimizationConfig:
    """Configuración del algoritmo de optimización"""
    
    # GRASP
    GRASP_CONFIG = {
        'max_iterations': 100,
        'alpha': 0.3,           # Factor aleatorio construcción golosa
        'time_limit': 300,      # 5 minutos máximo
        'improvement_threshold': 0.001  # Umbral mejora mínima
    }
    
    # Simulated Annealing
    ANNEALING_CONFIG = {
        'initial_temp': 1000,
        'cooling_rate': 0.95,
        'min_temp': 0.01,
        'max_swaps_per_temp': 50
    }
    
    # Clasificación de partidos
    MATCH_CLASSIFICATION = {
        'ancla_threshold': 0.60,        # Prob > 60% = Ancla
        'divisor_min': 0.40,           # 40% < Prob < 60% = Divisor
        'divisor_max': 0.60,
        'volatility_threshold': 0.15    # Cambio momios > 15% = Volatil
    }
    
    # Estrategias predefinidas
    ESTRATEGIAS = {
        'conservadora': {
            'pr11_objetivo': 0.10,
            'max_concentracion': 0.65,
            'distribucion': [0.45, 0.35, 0.20]  # Más locales
        },
        'balanceada': {
            'pr11_objetivo': 0.13,
            'max_concentracion': 0.70,
            'distribucion': [0.38, 0.29, 0.33]  # Histórica
        },
        'agresiva': {
            'pr11_objetivo': 0.16,
            'max_concentracion': 0.75,
            'distribucion': [0.30, 0.25, 0.45]  # Más visitantes
        }
    }

# ===== CONFIGURACIÓN DE BASE DE DATOS =====

class DatabaseConfig:
    """Configuración de base de datos (opcional)"""
    
    # PostgreSQL
    PG_HOST = os.getenv("PG_HOST", "localhost")
    PG_PORT = int(os.getenv("PG_PORT", "5432"))
    PG_DB = os.getenv("PG_DB", "progol")
    PG_USER = os.getenv("PG_USER", "progol_admin")
    PG_PASSWORD = os.getenv("PG_PASSWORD", "changeme")
    
    @classmethod
    def get_connection_string(cls) -> str:
        """Generar string de conexión PostgreSQL"""
        return (f"postgresql://{cls.PG_USER}:{cls.PG_PASSWORD}@"
                f"{cls.PG_HOST}:{cls.PG_PORT}/{cls.PG_DB}")

# ===== CONFIGURACIÓN DE STREAMLIT =====

class StreamlitConfig:
    """Configuración del dashboard Streamlit"""
    
    # Configuración del servidor
    PORT = int(os.getenv("STREAMLIT_PORT", "8501"))
    REFRESH_INTERVAL = int(os.getenv("DASHBOARD_REFRESH_INTERVAL", "5"))
    
    # Configuración de caché
    CACHE_TTL = 300  # 5 minutos
    
    # Límites de archivos
    MAX_FILE_SIZE_MB = 200
    ALLOWED_EXTENSIONS = ['.csv', '.pdf', '.json', '.xlsx']
    
    # Tema y colores
    THEME = {
        'primary_color': '#2ecc71',
        'background_color': '#ffffff',
        'secondary_background_color': '#f0f2f6',
        'text_color': '#262730'
    }

# ===== CONFIGURACIÓN DE LOGS Y ALERTAS =====

class LoggingConfig:
    """Configuración de logging y alertas"""
    
    # Configuración de logs
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE = "logs/progol_engine.log"
    
    # Alertas
    ENABLE_ALERTS = os.getenv("ENABLE_ALERTS", "false").lower() == "true"
    SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")
    EMAIL_NOTIFIER = os.getenv("EMAIL_NOTIFIER", "")
    
    # Umbrales para alertas
    ALERT_THRESHOLDS = {
        'pr11_bajo': 0.05,      # Pr[≥11] muy bajo
        'pr11_alto': 0.20,      # Pr[≥11] sospechosamente alto
        'tiempo_pipeline': 600,  # Pipeline > 10 min
        'error_rate': 0.10      # > 10% errores en simulación
    }

# ===== VARIABLES GLOBALES CONVENIENTES =====

# Jornada actual
JORNADA_ID = os.getenv("JORNADA_ID", "2283")

# Configuración de partidos global
PARTIDOS_CONFIG = PartidosConfig()

# Archivos
FILES = FileConfig()

# Pipeline
PIPELINE = PipelineConfig()

# Bayesiano
BAYESIAN = BayesianConfig()

# Optimización
OPTIMIZATION = OptimizationConfig()

# Base de datos
DATABASE = DatabaseConfig()

# Streamlit
STREAMLIT = StreamlitConfig()

# Logging
LOGGING = LoggingConfig()

# ===== FUNCIONES UTILITARIAS =====

def get_current_config() -> Dict:
    """Obtener configuración actual completa"""
    return {
        'project': {
            'name': PROJECT_NAME,
            'version': VERSION,
            'description': DESCRIPTION
        },
        'partidos': {
            'regulares': PARTIDOS_CONFIG.PARTIDOS_REGULARES,
            'revancha_max': PARTIDOS_CONFIG.PARTIDOS_REVANCHA_MAX,
            'total_max': PARTIDOS_CONFIG.PARTIDOS_TOTAL_MAX
        },
        'jornada': JORNADA_ID,
        'pipeline': {
            'n_quinielas': PIPELINE.N_QUINIELAS,
            'costo_boleto': PIPELINE.COSTO_BOLETO,
            'premio_cat2': PIPELINE.PREMIO_CAT2
        },
        'files': {
            'raw_path': FILES.RAW_DATA_PATH,
            'processed_path': FILES.PROCESSED_DATA_PATH,
            'outputs_path': FILES.OUTPUTS_PATH
        }
    }

def save_config_to_file(config: Dict, filepath: str):
    """Guardar configuración a archivo JSON"""
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

def load_config_from_file(filepath: str) -> Dict:
    """Cargar configuración desde archivo JSON"""
    with open(filepath, 'r') as f:
        return json.load(f)

def validate_environment() -> Dict[str, bool]:
    """Validar configuración del entorno"""
    validations = {}
    
    # Verificar directorios
    validations['directories'] = all([
        Path(FILES.RAW_DATA_PATH).exists(),
        Path(FILES.PROCESSED_DATA_PATH).exists(),
        Path(FILES.OUTPUTS_PATH).exists()
    ])
    
    # Verificar archivos críticos
    critical_files = [
        'src/utils/config.py',
        'requirements.txt'
    ]
    validations['critical_files'] = all([
        Path(f).exists() for f in critical_files
    ])
    
    # Verificar variables de entorno
    required_env_vars = ['JORNADA_ID']
    validations['env_vars'] = all([
        os.getenv(var) is not None for var in required_env_vars
    ])
    
    return validations

def initialize_system():
    """Inicializar sistema - crear directorios y configuración básica"""
    # Crear directorios
    FILES.ensure_directories()
    
    # Crear logs directory
    Path(LOGS_DIR).mkdir(parents=True, exist_ok=True)
    
    # Guardar configuración actual
    config_path = f"{FILES.OUTPUTS_PATH}/current_config.json"
    save_config_to_file(get_current_config(), config_path)
    
    return True

# Inicializar automáticamente al importar
if __name__ != "__main__":
    try:
        initialize_system()
    except Exception as e:
        print(f"Warning: No se pudo inicializar completamente el sistema: {e}")

# ===== EXPORTS PRINCIPALES =====

__all__ = [
    'PROJECT_NAME', 'VERSION', 'DESCRIPTION',
    'JORNADA_ID',
    'PartidosConfig', 'FileConfig', 'PipelineConfig', 
    'BayesianConfig', 'OptimizationConfig', 'DatabaseConfig',
    'StreamlitConfig', 'LoggingConfig',
    'PARTIDOS_CONFIG', 'FILES', 'PIPELINE', 'BAYESIAN',
    'OPTIMIZATION', 'DATABASE', 'STREAMLIT', 'LOGGING',
    'get_current_config', 'validate_environment', 'initialize_system'
]