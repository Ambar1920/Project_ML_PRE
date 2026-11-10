# ADR-001: Selección del Algoritmo de Clasificación Supervisada (Random Forest)

## Estado
Aceptado

## Contexto
Para el proyecto de predicción de rendimiento estudiantil (Aprobado vs. Reprobado) en la Maestría en IA, Educación e Investigación, se requiere un modelo de aprendizaje supervisado capaz de procesar variables heterogéneas (atributos numéricos continuos como horas de estudio y calificaciones, y categóricos como educación parental y actividades extracurriculares). 

Los requerimientos del sistema exigen:
1. Capacidad para capturar relaciones no lineales e interacciones entre hábitos de estudio y rendimiento sin necesidad de asunciones paramétricas estrictas.
2. Alta robustez frente al sobreajuste (overfitting).
3. Interpretabilidad mediante métricas de importancia de características (*Feature Importance*) para justificar las recomendaciones pedagógicas ante directivos y docentes.
4. Baja complejidad de calibración e inferencia en tiempo real (< 100 ms) en aplicaciones web (Streamlit y navegadores).

Se evaluaron tres alternativas principales:
- **Alternativa A: Regresión Logística (L1/L2):** Muy rápida e interpretable, pero asume linealidad en el logit y tiene dificultades para capturar interacciones multivariadas complejas sin ingeniería manual exhaustiva de polinomios.
- **Alternativa B: Random Forest Classifier:** Ensamble no paramétrico de árboles de decisión con embolsado (*bagging*), resistente a outliers, manejo nativo de datos tabulares mixtos y cálculo de impureza de Gini para interpretabilidad.
- **Alternativa C: Gradient Boosting (XGBoost / LightGBM):** Excelente rendimiento competitivo, pero mayor propensión al sobreajuste en datasets con fuerte desbalance si no se cuenta con una calibración fina de regularización e hiperparámetros.

## Decisión
Se decide implementar **Random Forest Classifier** (`sklearn.ensemble.RandomForestClassifier`) con 100 estimadores (`n_estimators=100`), profundidad máxima controlada (`max_depth=5`), semilla determinística (`random_state=42`) y ponderación de clases balanceada (`class_weight='balanced'`).

## Consecuencias

### Positivas
- **Excelente generalización:** El promediado de 100 árboles reduce la varianza y garantiza estabilidad estadística en datos no observados.
- **Interpretabilidad directa:** Permite extraer de inmediato las variables más determinantes del éxito académico para alimentar la Ficha de Recomendación Pedagógica.
- **Bajo costo computacional:** Inferencia ultrarrápida compatible con microservicios y navegadores.

### Negativas / Mitigaciones
- **Tamaño del modelo:** Los ensambles de árboles ocupan mayor memoria que un vector de coeficientes de regresión lineal. Mitigación: se restringe la profundidad máxima a `max_depth=5`, lo que mantiene el modelo ligero (< 2 MB) y exportable.
- **Interpretabilidad local individual:** Aunque ofrece importancia global, requiere aproximaciones heurísticas para explicaciones punto a punto. Mitigación: se construyó un motor prescriptivo basado en reglas pedagógicas que traduce las probabilidades en sugerencias didácticas comprensibles.

