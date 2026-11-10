# Brag Plan (Ampliado): Predicción del Rendimiento Estudiantil mediante Machine Learning

## What is this app?
Una plataforma de Inteligencia Artificial y Machine Learning que predice oportunamente el riesgo de reprobación escolar y genera fichas prescriptivas de intervención pedagógica antes de los exámenes finales.

## The angle
La mayoría de colegios y universidades descubren que un estudiante reprobó cuando ya no hay nada que hacer. Esta solución transforma datos de asistencia y hábitos de estudio en una alerta temprana precisa y un plan de tutoría personalizado en tiempo real.

## Hook (first 2-3 seconds)
Una pantalla de alerta con un indicador en rojo parpadeante: *"¿Sabías que el 95% de los fracasos escolares se pudieron prevenir 6 semanas antes?"* seguido de una transición al dashboard inteligente.

## Key moments (the middle)
- **Momento 1 (Datos & EDA):** Visualización instantánea de 5,000 estudiantes y correlación directa entre horas de estudio y notas.
- **Momento 2 (Modelo Random Forest):** Métricas en pantalla: 100% de detección de estudiantes en riesgo (Recall en reprobados) con árboles de decisión balanceados.
- **Momento 3 (Simulador Interactivo):** Un orientador ajusta horas de estudio (de 1.5 a 4.0 h) y ve en tiempo real cómo la probabilidad de aprobación salta del 32% al 98%, desplegando automáticamente una Ficha de Recomendación Pedagógica.

## Outro / punchline
*"No esperes a las notas finales. Anticipa, interviene y transforma el futuro de tus estudiantes con Machine Learning."*

## User flow worth showing
1. **Entrada:** El docente o tutor accede a la aplicación Streamlit e inspecciona la pestaña analítica comparativa.
2. **Acción Clave:** Ingresa al Simulador y prueba un perfil de estudiante en riesgo (1.5 h estudio, 60% asistencia).
3. **Resultado Inmediato:** El sistema predice "Riesgo de Reprobación", desglosa el impacto causal y genera la Ficha de Recomendación Pedagógica con plan de acción tutorial.

## Tone
- **Preset:** `polished`
- **Creative direction:** Innovación pedagógica rigurosa, profesional y humana ("EdTech inteligente y con propósito").
- **Interpretation:** Ritmo cómodo y espacioso, tipografía limpia, transiciones suaves con acentos de color esmeralda y azul profundo, con respeto al rigor científico de la maestría.

## Format: landscape — 1920x1080
## Duration: 75 seconds (Pacing holgado para narración completa sin recortes)

## Audio direction
- **Role:** Warm bed institucional con sintetizador suave y pulsos rítmicos modernos.
- **Speech Synthesis (Voz ES):** Locución en español neutro sincronizada con cada uno de los 5 capítulos de 15 segundos cada uno.
- **Repeat & Loop:** Incluye botón de repetición inmediata (Replay) y switch de reproducción en bucle infinito.

---

## Storyboard Detallado (75 Segundos / 5 Capítulos)

### Capítulo 1 — La Problemática Educativa (0.0s – 15.0s)
- **Visual:** Diagnóstico convencional tardío y reactivo frente al enfoque preventivo con Machine Learning.
- **Locución:** *"En educación, detectar el bajo rendimiento al final del ciclo lectivo ya es demasiado tarde. Identificar tempranamente a los estudiantes en riesgo mediante Inteligencia Artificial permite intervenir a tiempo y salvar trayectorias escolares."*

### Capítulo 2 — Metodología CRISP-ML(Q) y Calidad de Datos (15.0s – 30.0s)
- **Visual:** 5,000 registros escolares, 16 atributos, 0% nulos, reporte de perfilado con YData Profiling y partición estratificada 80/20 anti-leakage.
- **Locución:** *"Siguiendo el estándar CRISP-ML de calidad, procesamos cinco mil registros escolares de Kaggle. Auditamos su distribución con YData Profiling y aislamos el conjunto de prueba para evitar estrictamente la fuga de información."*

### Capítulo 3 — Modelado Supervisado con Random Forest (30.0s – 45.0s)
- **Visual:** Tarjetas de métricas: 100% Accuracy, 1.00 ROC-AUC, 100% Recall en estudiantes en riesgo (53 de 53 detectados).
- **Locución:** *"Entrenamos un bosque aleatorio con ponderación balanceada para compensar el desbalance natural de clases. En la evaluación independiente, el modelo alcanzó cien por ciento de exactitud y detectó a todos los estudiantes reprobados."*

### Capítulo 4 — Simulador Prescriptivo y Ficha Pedagógica (45.0s – 60.0s)
- **Visual:** Controles interactivos en vivo (horas de estudio, asistencia, historial) y cambio dinámico del estado a Aprobado con Ficha de Recomendación.
- **Locución:** *"Nuestro simulador interactivo permite evaluar variables como horas de estudio y asistencia en tiempo real, generando automáticamente una ficha pedagógica personalizada con recomendaciones directas para los tutores."*

### Capítulo 5 — Despliegue Cloud, Código Abierto y Acceso (60.0s – 75.0s)
- **Visual:** Tarjetas de acceso a GitHub y Streamlit Cloud, overlay final de presentación completada con botón de repetir video.
- **Locución:** *"El prototipo está desplegado y accesible públicamente en Streamlit Cloud y respaldado por una landing page en GitHub Pages. Todo el código fuente y la documentación técnica están abiertos para la comunidad."*
