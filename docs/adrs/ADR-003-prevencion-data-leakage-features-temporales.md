# ADR-003: Prevención de Fuga de Información (Data Leakage) y Distinción de Features Temporales

## Estado
Aceptado

## Contexto
Durante el análisis exploratorio y la ejecución de `Project_ML_PRE.ipynb`, se identificaron dos fenómenos críticos en el dataset de rendimiento escolar:
1. **Fuga en Preprocesamiento:** Si el escalamiento estadístico (`StandardScaler`) o la codificación (`LabelEncoder`) se ajustan sobre todo el conjunto de datos antes de la división de entrenamiento/prueba, información del conjunto de test se filtra a train, falseando la capacidad de generalización.
2. **Fuga por Variables Diagnósticas Post-Facto (Target Proxy Leakage):** El dataset contiene variables como `Final_Percentage` (porcentaje final) y notas de exámenes que se calculan al terminar el ciclo escolar. Dado que la regla de negocio para definir la columna `Pass_Fail` proviene directamente del umbral de `Final_Percentage`, incluir estas columnas como predictoras conduce a un modelo trivial con métricas perfectas (1.0000) que no resulta útil para la **alerta temprana preventiva**.

Un sistema de intervención pedagógica oportuna necesita predecir el riesgo **al inicio o a mitad del curso**, cuando el porcentaje final aún no existe.

## Decisión
Se establecen dos directrices metodológicas obligatorias:
1. **Aislamiento Estricto de Transformaciones:**
   - La partición de datos (`train_test_split`) es el primer paso metodológico.
   - `StandardScaler.fit()` se ejecuta única y exclusivamente sobre `X_train`.
   - `X_test` y los nuevos datos de inferencia se procesan exclusivamente con `.transform()`.
2. **Arquitectura de Modelado Dual:**
   - **Modelo Diagnóstico / Cuaderno Colab:** Evalúa la consistencia de las 15 variables consolidadas tal como vienen en el cuaderno base `Project_ML_PRE.ipynb`.
   - **Módulo Predictivo de Alerta Temprana (Simulador y Streamlit):** Privilegia y pondera las variables modificables tempranas (`Study_Hours_Per_Day`, `Attendance_Percentage`, `Previous_Year_Score` y `Extracurricular_Activities`), aislando variables causales de aquellas meramente derivadas.

## Consecuencias

### Positivas
- Se garantiza la integridad científica y ética del proyecto para la sustentación de la maestría.
- El prototipo interactivo simula un escenario pedagógico realista aplicable en colegios e IES.
- Se documenta con claridad metodológica por qué el modelo del cuaderno alcanza 1.00 de Accuracy y cómo se traduce operativamente a la práctica docente.

### Negativas / Mitigaciones
- Requiere explicar explícitamente en la sustentación y en la documentación técnica la diferencia entre modelado retrospectivo y predictivo en tiempo real. Se abordan detalladamente estas diferencias en la metodología CRISP-ML y en la landing page.

