#!/bin/bash

# =============================================================================
# PROGOL ENGINE - INSTALADOR AUTOMÁTICO PARA UBUNTU
# =============================================================================
# Automatiza la instalación completa desde 0 en Ubuntu
# Uso: bash install_progol_ubuntu.sh [opciones]
# =============================================================================

set -e  # Salir si cualquier comando falla

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # Sin color

# Variables globales
PYTHON_VERSION="3.11"
AIRFLOW_VERSION="2.7.3"
PROJECT_NAME="progol-engine"
DEFAULT_REPO_URL="https://github.com/CMCFame/pgdl.git"
INSTALL_LOG="/tmp/progol_install.log"
START_TIME=$(date +%s)

# Funciones de utilidad
print_header() {
    echo -e "\n${BLUE}================================${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}================================${NC}\n"
}

print_step() {
    echo -e "${CYAN}➤ $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

log_command() {
    echo "$(date): $1" >> "$INSTALL_LOG"
    eval "$1" 2>&1 | tee -a "$INSTALL_LOG"
}

check_command() {
    if command -v "$1" >/dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

show_help() {
    cat << EOF
PROGOL ENGINE - Instalador Automático Ubuntu

USO:
    bash install_progol_ubuntu.sh [opciones]

OPCIONES:
    -h, --help              Mostrar esta ayuda
    -s, --skip-system       Saltar instalación de paquetes del sistema
    -c, --clone-repo        Clonar repositorio (default: si no existe pgdl/)
    -u, --repo-url URL      URL del repositorio (default: https://github.com/CMCFame/pgdl.git)
    -d, --directory DIR     Directorio donde instalar (default: pgdl)
    --skip-airflow         Saltar configuración de Airflow
    --skip-verification    Saltar verificación final

EJEMPLOS:
    # Instalación completa (clona automáticamente si no existe)
    bash install_progol_ubuntu.sh

    # Instalación con URL personalizada
    bash install_progol_ubuntu.sh -u https://github.com/otro-usuario/pgdl.git

    # Saltar paquetes del sistema (si ya están instalados)
    bash install_progol_ubuntu.sh -s

    # Solo dependencias (código ya presente)
    bash install_progol_ubuntu.sh --skip-system -c false

EOF
}

# Parsear argumentos
SKIP_SYSTEM=false
CLONE_REPO=false
REPO_URL="$DEFAULT_REPO_URL"
INSTALL_DIR="pgdl"
SKIP_AIRFLOW=false
SKIP_VERIFICATION=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -s|--skip-system)
            SKIP_SYSTEM=true
            shift
            ;;
        -c|--clone-repo)
            CLONE_REPO=true
            shift
            ;;
        -u|--repo-url)
            REPO_URL="$2"
            CLONE_REPO=true
            shift 2
            ;;
        -d|--directory)
            INSTALL_DIR="$2"
            shift 2
            ;;
        --skip-airflow)
            SKIP_AIRFLOW=true
            shift
            ;;
        --skip-verification)
            SKIP_VERIFICATION=true
            shift
            ;;
        *)
            echo "Opción desconocida: $1"
            show_help
            exit 1
            ;;
    esac
done

# Función principal de instalación
main() {
    print_header "PROGOL ENGINE - INSTALACIÓN AUTOMÁTICA"
    
    echo -e "Configuración:"
    echo -e "  📁 Directorio: ${CYAN}$INSTALL_DIR${NC}"
    echo -e "  🐍 Python: ${CYAN}$PYTHON_VERSION${NC}"
    echo -e "  🌊 Airflow: ${CYAN}$AIRFLOW_VERSION${NC}"
    echo -e "  🔗 Repo: ${CYAN}$REPO_URL${NC}"
    echo -e "  📝 Log: ${CYAN}$INSTALL_LOG${NC}"
    
    echo -e "\nPresiona ENTER para continuar o Ctrl+C para cancelar..."
    read -r
    
    # Inicializar log
    echo "=== PROGOL ENGINE INSTALLATION LOG ===" > "$INSTALL_LOG"
    echo "Started: $(date)" >> "$INSTALL_LOG"
    
    # Ejecutar pasos
    step_1_system_packages
    step_2_clone_repository
    step_3_python_environment
    step_4_python_dependencies
    step_5_project_setup
    step_6_airflow_setup
    step_7_verification
    
    show_completion_summary
}

step_1_system_packages() {
    if [[ "$SKIP_SYSTEM" == true ]]; then
        print_warning "Saltando instalación de paquetes del sistema"
        return 0
    fi
    
    print_header "PASO 1: PAQUETES DEL SISTEMA"
    
    print_step "Actualizando índice de paquetes..."
    log_command "sudo apt update"
    
    print_step "Instalando dependencias del sistema..."
    log_command "sudo apt install -y \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-venv \
        python${PYTHON_VERSION}-dev \
        build-essential \
        gcc g++ gfortran \
        libatlas-base-dev \
        liblapack-dev \
        libblas-dev \
        libssl-dev \
        libffi-dev \
        postgresql-client \
        git curl wget \
        htop tree"
    
    print_success "Paquetes del sistema instalados"
}

step_2_clone_repository() {
    print_header "PASO 2: REPOSITORIO"
    
    # Si el directorio ya existe y no queremos clonar
    if [[ -d "$INSTALL_DIR" && "$CLONE_REPO" == false ]]; then
        print_warning "Directorio $INSTALL_DIR ya existe, usando código existente"
        cd "$INSTALL_DIR"
        return 0
    fi
    
    # Si el directorio no existe, clonar automáticamente
    if [[ ! -d "$INSTALL_DIR" ]]; then
        print_step "Directorio no existe, clonando repositorio automáticamente..."
        CLONE_REPO=true
    fi
    
    if [[ "$CLONE_REPO" == true ]]; then
        if [[ -d "$INSTALL_DIR" ]]; then
            print_warning "Directorio $INSTALL_DIR ya existe"
            echo -e "¿Eliminar y clonar de nuevo? (y/N): "
            read -r response
            if [[ "$response" =~ ^[Yy]$ ]]; then
                rm -rf "$INSTALL_DIR"
            else
                print_warning "Usando directorio existente"
                cd "$INSTALL_DIR"
                return 0
            fi
        fi
        
        print_step "Clonando repositorio desde $REPO_URL..."
        log_command "git clone $REPO_URL $INSTALL_DIR"
        cd "$INSTALL_DIR"
        print_success "Repositorio clonado en $INSTALL_DIR"
    else
        print_warning "Saltando clonado de repositorio"
        if [[ ! -d "$INSTALL_DIR" ]]; then
            print_error "Directorio $INSTALL_DIR no existe y no se especificó clonar"
            exit 1
        fi
        cd "$INSTALL_DIR"
    fi
}

step_3_python_environment() {
    print_header "PASO 3: ENTORNO VIRTUAL PYTHON"
    
    # Verificar Python
    if ! check_command "python${PYTHON_VERSION}"; then
        print_error "Python ${PYTHON_VERSION} no encontrado"
        exit 1
    fi
    
    PYTHON_CMD="python${PYTHON_VERSION}"
    PYTHON_ACTUAL=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2)
    print_step "Python detectado: $PYTHON_ACTUAL"
    
    # Crear entorno virtual
    if [[ -d "venv" ]]; then
        print_warning "Entorno virtual ya existe"
        echo -e "¿Recrear entorno virtual? (y/N): "
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            rm -rf venv
        else
            print_warning "Usando entorno virtual existente"
            source venv/bin/activate
            return 0
        fi
    fi
    
    print_step "Creando entorno virtual..."
    log_command "$PYTHON_CMD -m venv venv"
    
    print_step "Activando entorno virtual..."
    source venv/bin/activate
    
    print_step "Actualizando pip..."
    log_command "pip install --upgrade pip setuptools wheel"
    
    print_success "Entorno virtual configurado"
}

step_4_python_dependencies() {
    print_header "PASO 4: DEPENDENCIAS PYTHON"
    
    # Verificar que estamos en el entorno virtual
    if [[ "$VIRTUAL_ENV" == "" ]]; then
        print_error "Entorno virtual no activado"
        exit 1
    fi
    
    print_step "Instalando dependencias básicas..."
    log_command "pip install pandas==2.1.4 numpy==1.24.3 scipy==1.11.1"
    
    print_step "Instalando visualización..."
    log_command "pip install matplotlib==3.7.2 seaborn==0.12.2 plotly==5.15.0"
    
    print_step "Instalando Streamlit..."
    log_command "pip install streamlit==1.28.1 altair==5.0.1"
    
    print_step "Instalando Machine Learning..."
    log_command "pip install scikit-learn==1.3.0 statsmodels==0.14.0 joblib==1.3.1"
    
    print_step "Instalando base de datos..."
    log_command "pip install psycopg2-binary==2.9.7 sqlalchemy==2.0.23"
    
    print_step "Instalando utilidades..."
    log_command "pip install python-dotenv==1.0.0 click==8.1.7 tqdm==4.66.1 requests==2.31.0 beautifulsoup4==4.12.2"
    
    print_step "Instalando PyArrow..."
    log_command "pip install pyarrow==13.0.0"
    
    print_step "Instalando Airflow (esto puede tardar)..."
    CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"
    if ! log_command "pip install 'apache-airflow==${AIRFLOW_VERSION}' --constraint '${CONSTRAINT_URL}'"; then
        print_warning "Airflow con constraints falló, intentando sin constraints..."
        log_command "pip install apache-airflow==${AIRFLOW_VERSION}"
    fi
    
    print_step "Instalando providers de Airflow..."
    log_command "pip install apache-airflow-providers-postgres==5.6.0"
    
    print_step "Instalando modelos bayesianos..."
    if ! log_command "pip install cmdstanpy==1.2.0"; then
        print_warning "cmdstanpy falló, intentando pystan..."
        if ! log_command "pip install pystan==3.7.0"; then
            print_warning "Modelos bayesianos no instalados - continuando sin ellos"
        fi
    fi
    
    print_step "Instalando herramientas de desarrollo..."
    log_command "pip install pytest==7.4.3 black==23.11.0 flake8==6.1.0"
    
    print_step "Instalando Jupyter..."
    log_command "pip install jupyter==1.0.0 notebook==7.0.6 ipywidgets==8.1.1"
    
    # Instalar desde requirements.txt si existe
    if [[ -f "requirements.txt" ]]; then
        print_step "Instalando desde requirements.txt..."
        log_command "pip install -r requirements.txt"
    fi
    
    print_success "Todas las dependencias instaladas"
}

step_5_project_setup() {
    print_header "PASO 5: CONFIGURACIÓN DEL PROYECTO"
    
    # Ejecutar setup.py si existe
    if [[ -f "setup.py" ]]; then
        print_step "Ejecutando setup del proyecto..."
        log_command "python setup.py"
    else
        print_warning "setup.py no encontrado, configurando manualmente..."
        
        # Crear directorios básicos
        mkdir -p data/{raw,processed,dashboard,reports,json_previas}
        mkdir -p models/{poisson,bayes}
        mkdir -p logs
        mkdir -p .streamlit
        
        # Crear .env desde template si existe
        if [[ -f ".env.template" ]]; then
            cp .env.template .env
            print_step "Archivo .env creado desde template"
        fi
    fi
    
    # Configurar permisos de scripts
    if [[ -d "scripts" ]]; then
        chmod +x scripts/*.sh 2>/dev/null || true
        chmod +x scripts/*.py 2>/dev/null || true
    fi
    
    print_success "Proyecto configurado"
}

step_6_airflow_setup() {
    if [[ "$SKIP_AIRFLOW" == true ]]; then
        print_warning "Saltando configuración de Airflow"
        return 0
    fi
    
    print_header "PASO 6: CONFIGURACIÓN DE AIRFLOW"
    
    # Configurar AIRFLOW_HOME
    export AIRFLOW_HOME="$(pwd)/airflow"
    echo "export AIRFLOW_HOME=$(pwd)/airflow" >> ~/.bashrc
    
    print_step "Inicializando base de datos de Airflow..."
    log_command "airflow db init"
    
    print_step "Creando usuario admin de Airflow..."
    log_command "airflow users create \
        --username admin \
        --firstname Admin \
        --lastname User \
        --role Admin \
        --email admin@progol.com \
        --password admin123"
    
    print_step "Configurando DAGs..."
    if [[ -d "airflow/dags" ]]; then
        # Los DAGs ya deberían estar en su lugar
        print_success "DAGs encontrados en airflow/dags/"
    fi
    
    print_success "Airflow configurado"
    print_warning "Para iniciar Airflow manualmente:"
    echo -e "  ${CYAN}airflow webserver --port 8080 &${NC}"
    echo -e "  ${CYAN}airflow scheduler &${NC}"
}

step_7_verification() {
    if [[ "$SKIP_VERIFICATION" == true ]]; then
        print_warning "Saltando verificación"
        return 0
    fi
    
    print_header "PASO 7: VERIFICACIÓN DE INSTALACIÓN"
    
    print_step "Verificando imports críticos..."
    python -c "
import pandas as pd
import numpy as np
import streamlit as st
import sklearn
import matplotlib.pyplot as plt
print('✅ Imports básicos OK')
"
    
    # Ejecutar script de verificación si existe
    if [[ -f "scripts/verify_installation.py" ]]; then
        print_step "Ejecutando verificación completa..."
        python scripts/verify_installation.py
    fi
    
    print_step "Verificando estructura de directorios..."
    required_dirs=("data/raw" "data/processed" "models" "src" "streamlit_app")
    for dir in "${required_dirs[@]}"; do
        if [[ -d "$dir" ]]; then
            echo -e "  ${GREEN}✅${NC} $dir"
        else
            echo -e "  ${RED}❌${NC} $dir"
        fi
    done
    
    print_success "Verificación completada"
}

show_completion_summary() {
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    MINUTES=$((DURATION / 60))
    SECONDS=$((DURATION % 60))
    
    print_header "🎉 INSTALACIÓN COMPLETADA 🎉"
    
    echo -e "${GREEN}✅ Progol Engine (PGDL) instalado exitosamente${NC}"
    echo -e "⏱️  Tiempo total: ${CYAN}${MINUTES}m ${SECONDS}s${NC}"
    echo -e "📁 Ubicación: ${CYAN}$(pwd)${NC}"
    echo -e "📝 Log completo: ${CYAN}$INSTALL_LOG${NC}"
    
    echo -e "\n${PURPLE}🚀 PRÓXIMOS PASOS:${NC}"
    echo -e "1. Activar entorno virtual: ${CYAN}source venv/bin/activate${NC}"
    echo -e "2. Configurar .env con tus parámetros"
    echo -e "3. Iniciar dashboard: ${CYAN}streamlit run streamlit_app/dashboard.py${NC}"
    echo -e "4. Ejecutar pipeline: ${CYAN}make run-all${NC}"
    
    if [[ "$SKIP_AIRFLOW" != true ]]; then
        echo -e "\n${PURPLE}🌊 AIRFLOW:${NC}"
        echo -e "• Web UI: ${CYAN}http://localhost:8080${NC}"
        echo -e "• Usuario: ${CYAN}admin${NC} / Contraseña: ${CYAN}admin123${NC}"
        echo -e "• Iniciar: ${CYAN}airflow webserver --port 8080 & airflow scheduler &${NC}"
    fi
    
    echo -e "\n${PURPLE}📚 COMANDOS ÚTILES:${NC}"
    echo -e "• Tests: ${CYAN}make test${NC}"
    echo -e "• Linting: ${CYAN}make lint${NC}"
    echo -e "• Pipeline ETL: ${CYAN}make run-etl${NC}"
    echo -e "• Ayuda: ${CYAN}make help${NC}"
    
    echo -e "\n${GREEN}¡Progol Engine listo para usar! 🎯${NC}"
}

# Manejo de errores
trap 'print_error "Instalación interrumpida"; exit 1' INT TERM

# Verificar que estamos en Ubuntu
if ! grep -q "Ubuntu" /etc/os-release 2>/dev/null; then
    print_warning "Este script está optimizado para Ubuntu"
    echo -e "¿Continuar anyway? (y/N): "
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Ejecutar instalación
main "$@"