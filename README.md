# Sistema de Predicción de DRL en Angiografía

Sistema de Machine Learning para predecir la excedencia de Niveles de Referencia Diagnósticos (DRL) en procedimientos de angiografía.

## Descripción

Este proyecto utiliza técnicas de ML para identificar procedimientos que tienen alta probabilidad de exceder el P75 (DRL), permitiendo ajustes preventivos.

## Instalación

```bash
# 1. Clonar repositorio
git clone [URL]
cd angiografia_drl_project

# 2. Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt
