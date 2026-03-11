Manual de Usuario

Introducción

Este sistema permite predecir el riesgo de que un procedimiento de angiografía exceda los Niveles de Referencia Diagnósticos (DRL) utilizando un modelo de Machine Learning basado en XGBoost.
La aplicación cuenta con tres componentes principales:

API REST (FastAPI): permite realizar predicciones mediante solicitudes HTTP.
Dashboard interactivo (Streamlit): interfaz gráfica para ingresar datos del paciente y visualizar resultados.
MLflow: plataforma para visualizar experimentos y métricas de los modelos entrenados.

El sistema está diseñado para apoyar la toma de decisiones clínicas y contribuir a la optimización de la dosis radiológica en procedimientos de angiografía. 


1. Acceso al sistema

Una vez que el sistema está instalado y ejecutándose, se puede acceder a sus componentes desde el navegador web.

Los servicios disponibles son:

Servicio	URL
API FastAPI	http://localhost:8000
Documentación API	http://localhost:8000/docs
Dashboard	http://localhost:8501
MLflow	http://localhost:5000

2. Uso del Dashboard

El Dashboard desarrollado con Streamlit es la forma más sencilla de utilizar el sistema.
Acceso
Abrir el navegador y acceder a:
http://localhost:8501

-Ingreso de datos

El usuario debe ingresar las siguientes variables del paciente:
Tipo de procedimiento
Edad
Peso
PKA (Producto Kerma-Área)
Ka,r (Kerma en aire)
Tiempo de fluoroscopia

Estas variables corresponden a los factores clínicos y operativos utilizados por el modelo para realizar la predicción. 

-Predicción del riesgo

Una vez ingresados los datos, el sistema ejecuta el modelo y muestra el resultado mediante un indicador visual tipo semáforo:

Nivel	Color	Interpretación
Bajo	Verde	Procedimiento dentro de niveles seguros
Moderado	Amarillo	Riesgo moderado
Alto	Rojo	Probable excedencia del DRL

Esto permite identificar de forma rápida los casos con mayor riesgo radiológico.

3. Uso de la API

La API permite integrar el modelo con otros sistemas o realizar predicciones programáticamente.

Acceso a la documentación
Abrir en el navegador:
http://localhost:8000/docs

Aquí se encuentra la interfaz Swagger, que permite probar los endpoints.
Endpoint principal
POST /predict

Este endpoint recibe los datos del paciente en formato JSON.

Ejemplo de solicitud
{
  "tipo_procedimiento": "Coronariografia",
  "edad": 65,
  "peso": 78,
  "pka_gycm2": 60,
  "kar_mgy": 850,
  "tiempo_fluoroscopia_min": 12
}
Respuesta del sistema

El sistema devuelve:
Predicción del modelo
Probabilidad estimada
Nivel de riesgo

4. Visualización de experimentos con MLflow

El sistema incluye MLflow para analizar los experimentos de entrenamiento del modelo.
Acceso
Abrir:

http://localhost:5000

En esta interfaz se pueden visualizar:
Modelos entrenados
Métricas de desempeño
Parámetros utilizados
Comparación entre algoritmos
El modelo seleccionado para producción fue XGBoost, que alcanzó un accuracy cercano al 99% y un ROC-AUC de 0.997. 


5. Flujo típico de uso

El flujo normal para utilizar el sistema es:
Ejecutar los servicios del proyecto.
Abrir el Dashboard en Streamlit.
Ingresar los datos del paciente.
Obtener la predicción del riesgo radiológico.

Analizar los resultados para apoyar la toma de decisiones clínicas.

6. Consideraciones

El sistema es una herramienta de apoyo a la decisión clínica, no reemplaza el criterio médico.
Las predicciones dependen de la calidad de los datos ingresados.
El modelo fue entrenado con datos históricos de procedimientos de angiografía, por lo que se recomienda validar su uso en otros contextos clínicos.
