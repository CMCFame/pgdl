# Guía de Actualización - Progol Engine v2

## 🎯 Nuevas Funcionalidades

### Soporte para Revancha
- Ahora soporta hasta 21 partidos (14 regulares + 7 revancha)
- Configuración flexible del número de partidos

### UI Mejorada
- Dashboard más intuitivo
- Resultados disponibles directamente en la app
- Mejor gestión de archivos

### Optimizador Avanzado
- Algoritmo GRASP mejorado
- Arquitectura Core + Satélites optimizada
- Simulación Monte Carlo más precisa

## 📋 Pasos de Migración

### 1. Backup Completado
✅ Tu instalación anterior ha sido respaldada automáticamente

### 2. Archivos Nuevos Creados
- `run_progol_engine_v2.py` - Pipeline principal
- `streamlit_app/dashboard_v2_migration.py` - Dashboard mejorado
- `src/utils/config_v2.py` - Configuración actualizada

### 3. Próximos Pasos

#### Instalar Nuevas Dependencias
```bash
pip install -r requirements.txt
```

#### Probar Pipeline v2
```bash
python run_progol_engine_v2.py --jornada 2283 --help
```

#### Migrar Dashboard
1. Revisar `streamlit_app/dashboard_v2_migration.py`
2. Integrar configuraciones personalizadas
3. Probar con: `streamlit run streamlit_app.py`

#### Migrar Datos Existentes
```bash
python scripts/migrate_to_v2.py
```

## 🔧 Configuración

### Variables de Entorno Nuevas
Agregar a tu archivo `.env`:
```
# Configuración v2
PARTIDOS_REGULARES=14
PARTIDOS_REVANCHA_MAX=7
STREAMLIT_V2_ENABLED=true
```

## ⚠️ Problemas Conocidos

### Dashboard Original
- El dashboard original seguirá funcionando
- Para nuevas funcionalidades, usar archivos _v2

### Datos Existentes
- Los datos existentes son compatibles
- Ejecutar migración para mejor rendimiento

## 📞 Soporte

Si encuentras problemas:
1. Revisar logs en `logs/progol_engine_v2.log`
2. Restaurar backup si es necesario
3. Consultar documentación técnica

---
*Generado automáticamente por el actualizador de Progol Engine v2*
