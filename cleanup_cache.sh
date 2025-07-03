#!/bin/bash
# cleanup_cache.sh - Script de limpieza completa

echo "🧹 Limpiando caché de Python y Streamlit..."

# 1. Limpiar archivos .pyc y __pycache__
echo "📁 Limpiando archivos Python compilados..."
find . -name "*.pyc" -delete
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# 2. Limpiar caché de Streamlit
echo "🔄 Limpiando caché de Streamlit..."
rm -rf ~/.streamlit/
rm -rf .streamlit/

# 3. Limpiar archivos temporales
echo "🗑️ Limpiando archivos temporales..."
find . -name "*.swp" -delete
find . -name "*~" -delete
find . -name ".DS_Store" -delete

# 4. Opcional: Limpiar archivos backup problemáticos
echo "📋 Limpiando archivos de backup..."
find . -name "*_bak" -type f -ls
echo "¿Quieres eliminar los archivos _bak? (y/n)"
read -r response
if [[ $response == "y" ]]; then
    find . -name "*_bak" -type f -delete
    echo "✅ Archivos _bak eliminados"
else
    echo "⏭️ Archivos _bak conservados"
fi

echo "✅ Limpieza completada!"
echo "🚀 Ahora reinicia Streamlit completamente"