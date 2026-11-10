# ADR-002: Estrategia para Mitigar el Desbalance Extremo de Clases

## Estado
Aceptado

## Contexto
El dataset de rendimiento estudiantil (`suvidyasonawane/student-performance-dataset`) presenta una distribución asimétrica en la variable objetivo `Pass_Fail`:
- **Clase Mayoritaria (Aprobado / Pass):** ~94.7% de los casos (4,735 estudiantes).
- **Clase Minoritaria (Reprobado / Fail):** ~5.3% de los casos (265 estudiantes).

En el ámbito educativo, un falso negativo (un estudiante en riesgo severo de reprobación clasificado erróneamente como aprobado) tiene un costo formativo mucho más alto que un falso positivo (un estudiante que aprueba pero es citado a tutorías preventivas). Un clasificador ingenuo optimizado únicamente para *Accuracy* alcanzaría un 94.7% de acierto prediciendo siempre "Aprobado", pero fracasaría en el objetivo central del proyecto: **detectar estudiantes en riesgo**.

Se analizaron tres alternativas:
1. **Submuestreo aleatorio (Random Undersampling):** Reducir la clase mayoritaria a 265 casos. Desventaja: pérdida crítica del 90% de la información y degradación estadística.
2. **Sobremuestreo sintético (SMOTE - Synthetic Minority Over-sampling Technique):** Generar ejemplos sintéticos en el espacio de características. Desventaja: riesgo de generar instancias irreales en datos discretos y categóricos codificados, además de complejizar el pipeline de inferencia.
3. **Ponderación de pesos de clase basada en frecuencia inversa (`class_weight='balanced'`):** Penalizar fuertemente en la función de pérdida los errores cometidos en la clase minoritaria sin alterar los datos originales.

## Decisión
Se adopta la estrategia de **pesos de clase balanceados (`class_weight='balanced'`)** tanto en el entrenamiento del modelo como en la partición estratificada (`stratify=y` en `train_test_split`).

Fórmula de ponderación aplicada:
$$w_j = \frac{N}{k \cdot n_j}$$
Donde:
- $N = 4000$ (tamaño de la muestra de entrenamiento).
- $k = 2$ (dos clases).
- $n_0 = 212$ (muestras reprobadas en train), asignando un peso $w_0 \approx 9.43$.
- $n_1 = 3788$ (muestras aprobadas en train), asignando un peso $w_1 \approx 0.53$.

## Consecuencias

### Positivas
- Se garantiza que el algoritmo preste máxima atención a los patrones asociados a la reprobación.
- No se introducen datos sintéticos que puedan inducir falsas correlaciones pedagógicas.
- Se mantiene el soporte empírico real del conjunto de prueba para auditorías rigurosas.
- El *Recall* de la clase "Reprobado" alcanza el 100% en la muestra de prueba evaluada.

### Negativas / Mitigaciones
- Posible incremento de falsos positivos en umbrales estándar (0.5). Mitigación: en la aplicación Streamlit se reporta la probabilidad continua ($0 - 100\%$) para permitir al docente calibrar el nivel de intervención según la capacidad tutorial de la institución.

