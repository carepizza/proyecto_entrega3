# Manual de Instalación

A continuación se describen los pasos para instalar y ejecutar el proyecto en un entorno local.

## 1. Clonar el repositorio
Primero se debe descargar el contenido del repositorio en el equipo local con el siguiente comando:

git clone https://github.com/paolaquisoboni-max/proyecto_entrega3.git

Luego se ingresa a la carpeta del proyecto:

cd proyecto_entrega3

## 2. Crear y activar un entorno virtual (opcional)
Se recomienda crear un entorno virtual para aislar las dependencias del proyecto y evitar conflictos con otras librerías instaladas en el sistema.

Para crearlo, ejecutar:

python -m venv venv

Para activarlo en Windows:

venv\Scripts\activate

Para activarlo en Linux o Mac:

source venv/bin/activate

## 3. Instalar las dependencias
Una vez activado el entorno virtual, se deben instalar las librerías necesarias para el proyecto. Primero se actualiza `pip` y luego se instalan las dependencias listadas en `requirements.txt`:

pip install --upgrade pip
pip install -r requirements.txt

## 4. Descargar archivos versionados con DVC
Este proyecto utiliza DVC para gestionar archivos de datos y modelos. Por ello, después de instalar las dependencias, se deben descargar los archivos versionados con el siguiente comando:

dvc pull

## 5. Ejecutar los servicios del proyecto
Después de completar la instalación, se pueden levantar los diferentes servicios incluidos en el proyecto.

### MLflow (opcional)
Para iniciar la interfaz de seguimiento de experimentos con MLflow, ejecutar:

mlflow ui --port 5000

Luego se puede acceder desde el navegador en:

http://localhost:5000

### API
Para ejecutar la API desarrollada con FastAPI, usar el siguiente comando:

uvicorn api.main:app --reload

La documentación interactiva de la API estará disponible en:

http://localhost:8000/docs

### Dashboard
Para ejecutar el dashboard desarrollado en Streamlit, usar:

streamlit run dashboard/app.py

Luego se podrá abrir en el navegador en:

http://localhost:8501
