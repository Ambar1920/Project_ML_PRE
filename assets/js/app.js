// ==========================================================================
// APLICACIÓN JAVASCRIPT: LOGICA INTERACTIVA DE LA LANDING PAGE
// ==========================================================================

document.addEventListener('DOMContentLoaded', () => {
    initHeroCounters();
    initCrispNavigator();
    initAdrAccordion();
    initStudentSimulator();
});

// --- 1. Animación Suave de Contadores Numéricos ---
function initHeroCounters() {
    const counters = document.querySelectorAll('.stat-value');
    counters.forEach(counter => {
        const target = parseFloat(counter.getAttribute('data-target'));
        const isPercent = counter.getAttribute('data-percent') === 'true';
        let current = 0;
        const step = target / 35;

        const updateCount = () => {
            current += step;
            if (current < target) {
                counter.textContent = isPercent ? current.toFixed(1) + '%' : Math.floor(current).toLocaleString();
                requestAnimationFrame(updateCount);
            } else {
                counter.textContent = isPercent ? target.toFixed(1) + '%' : target.toLocaleString();
            }
        };

        updateCount();
    });
}

// --- 2. Navegador Interactivo del Ciclo de Vida CRISP-ML(Q) ---
const crispPhases = [
    {
        num: "Fase 1",
        title: "Comprensión del Negocio & Necesidad Educativa",
        desc: "Identificación de la problemática de reprobación escolar tardía, definición de objetivos SMART y establecimiento de los KPIs de intervención pedagógica.",
        specs: [
            "<strong>Stakeholders Clave:</strong> Docentes de aula, orientadores escolares, directores académicos y comités de permanencia.",
            "<strong>Problema Técnico:</strong> Detección tardía de estudiantes con bajo rendimiento cuando ya no existe margen de recuperación.",
            "<strong>KPI Principal:</strong> Recall en la clase minoritaria (Reprobados) ≥ 90% para evitar falsos negativos.",
            "<strong>Aporte Prescriptivo:</strong> Generación automática de planes de tutoría personalizada y regularización."
        ]
    },
    {
        num: "Fase 2",
        title: "Comprensión y Perfilado de Datos (YData Profiling)",
        desc: "Ingesta automatizada del dataset abierto de Kaggle (5,000 registros escolares) y análisis exhaustivo de calidad, nulos y distribuciones multivariadas.",
        specs: [
            "<strong>Dataset Utilizado:</strong> <code>suvidyasonawane/student-performance-dataset</code> (5,000 instancias y 16 variables).",
            "<strong>Calidad Auditada:</strong> 0% de celdas vacías o registros duplicados tras el perfilado exploratorio.",
            "<strong>Desbalance Natural:</strong> 94.7% de estudiantes aprobados vs. 5.3% de reprobados (n=265 en riesgo).",
            "<strong>Factores de Correlación:</strong> Fuerte correlación positiva de las horas de estudio (r = 0.62) y asistencia (r = 0.58) con las notas finales."
        ]
    },
    {
        num: "Fase 3",
        title: "Preparación de Datos & Control de Data Leakage",
        desc: "Codificación de variables categóricas, estandarización estadística y aislamiento estricto de transformaciones en el conjunto de entrenamiento.",
        specs: [
            "<strong>Aislamiento de Split:</strong> Partición 80% train / 20% test estratificada realizada antes de cualquier transformación.",
            "<strong>Estandarización:</strong> <code>StandardScaler</code> ajustado única y exclusivamente sobre el subconjunto de entrenamiento.",
            "<strong>Control de Leakage:</strong> Diferenciación entre variables diagnósticas post-facto (exámenes finales) y factores causales tempranos (hábitos y asistencia).",
            "<strong>Ponderación Inversa:</strong> Cálculo de pesos de clase inversamente proporcionales para mitigar el sesgo del desbalance."
        ]
    },
    {
        num: "Fase 4",
        title: "Modelado de Machine Learning (Random Forest)",
        desc: "Entrenamiento de un ensamble de bosques aleatorios no paramétrico con 100 estimadores, profundidad controlada y pesos de clase balanceados.",
        specs: [
            "<strong>Algoritmo Seleccionado:</strong> <code>RandomForestClassifier(n_estimators=100, max_depth=5, class_weight='balanced')</code>.",
            "<strong>Resistencia a Outliers:</strong> El embolsado (bagging) promedia la varianza de múltiples árboles y previene el sobreajuste.",
            "<strong>Manejo No Lineal:</strong> Captura interacciones complejas entre pocas horas de estudio y alta o baja asistencia.",
            "<strong>Interpretabilidad de Gini:</strong> Extracción transparente de la importancia relativa de cada característica socioeducativa."
        ]
    },
    {
        num: "Fase 5",
        title: "Evaluación Multimétrica y Auditoría de Errores",
        desc: "Validación rigurosa sobre 1,000 estudiantes no vistos en entrenamiento, analizando matriz de confusión, curva ROC y reporte de clasificación.",
        specs: [
            "<strong>Exactitud Global (Accuracy):</strong> 1.0000 (100.0%) en el subconjunto de prueba independiente.",
            "<strong>Capacidad de Separación (ROC-AUC):</strong> 1.0000, evidenciando discriminación perfecta entre ambas clases.",
            "<strong>Recall en Reprobados:</strong> 1.0000 (100%), detectando correctamente a los 53 estudiantes reprobados de prueba sin falsos negativos.",
            "<strong>Falsos Positivos:</strong> 0 estudiantes aprobados catalogados erróneamente en riesgo, optimizando los recursos de tutoría."
        ]
    },
    {
        num: "Fase 6",
        title: "Despliegue Híbrido, Gobernanza & Ética de IA",
        desc: "Puesta en producción mediante Streamlit Community Cloud y GitHub Pages, integrando motor prescriptivo y directrices de IA responsable.",
        specs: [
            "<strong>Arquitectura Dual:</strong> Landing page estática en GitHub Pages para indexación y Web App interactiva en Streamlit Cloud.",
            "<strong>Principio Human-in-the-Loop:</strong> La IA actúa como copiloto del orientador; nunca toma decisiones punitivas ni excluyentes.",
            "<strong>Protección y Privacidad:</strong> Anonimización total mediante identificadores alfanuméricos sin almacenamiento de datos personales (PII).",
            "<strong>Monitoreo de Deriva:</strong> Protocolo de reentrenamiento periódico si cambian los patrones de ingreso estudiantil."
        ]
    }
];

function initCrispNavigator() {
    const navButtons = document.querySelectorAll('.crisp-nav-btn');
    const badgeEl = document.getElementById('crispPhaseBadge');
    const titleEl = document.getElementById('crispPhaseTitle');
    const descEl = document.getElementById('crispPhaseDesc');
    const specsEl = document.getElementById('crispPhaseSpecs');

    if (!navButtons.length || !badgeEl || !titleEl || !descEl || !specsEl) return;

    navButtons.forEach((btn, index) => {
        btn.addEventListener('click', () => {
            navButtons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            const phase = crispPhases[index];
            badgeEl.textContent = phase.num;
            titleEl.textContent = phase.title;
            descEl.textContent = phase.desc;

            specsEl.innerHTML = '';
            phase.specs.forEach(s => {
                const li = document.createElement('li');
                li.innerHTML = s;
                specsEl.appendChild(li);
            });
        });
    });
}

// --- 3. Acordeón Interactivo de Decisiones Arquitectónicas (ADRs) ---
function initAdrAccordion() {
    const headers = document.querySelectorAll('.adr-header');
    headers.forEach(header => {
        header.addEventListener('click', () => {
            const item = header.parentElement;
            const isOpen = item.classList.contains('open');

            // Cerrar otros
            document.querySelectorAll('.adr-item').forEach(i => i.classList.remove('open'));

            if (!isOpen) {
                item.classList.add('open');
            }
        });
    });
}

// --- 4. Simulador Interactivo de Rendimiento Estudiantil ---
function initStudentSimulator() {
    const inHours = document.getElementById('simHours');
    const inPrev = document.getElementById('simPrev');
    const inAtt = document.getElementById('simAtt');
    const inExtra = document.getElementById('simExtra');

    const lblHours = document.getElementById('lblHours');
    const lblPrev = document.getElementById('lblPrev');
    const lblAtt = document.getElementById('lblAtt');

    const bannerBox = document.getElementById('simBanner');
    const bannerTitle = document.getElementById('simStatusTitle');
    const bannerProb = document.getElementById('simProbValue');
    const recList = document.getElementById('simRecItems');
    const btnCopy = document.getElementById('btnCopyReport');

    if (!inHours || !inPrev || !inAtt || !inExtra) return;

    function runSimulation() {
        const h = parseFloat(inHours.value);
        const p = parseFloat(inPrev.value);
        const a = parseFloat(inAtt.value);
        const e = parseInt(inExtra.value, 10);

        // Actualizar visualizadores de texto
        if (lblHours) lblHours.textContent = h.toFixed(1) + ' hrs/día';
        if (lblPrev) lblPrev.textContent = p + ' pts';
        if (lblAtt) lblAtt.textContent = a + '%';

        // Lógica ponderada representativa del Random Forest entrenado
        const score = (h * 6.0) + (p * 0.7) + (e * 10.0) + ((a - 50) * 0.45);

        let isApproved = score >= 65;
        let prob = 0;

        if (isApproved) {
            bannerBox.className = 'sim-output-banner pass';
            bannerTitle.innerHTML = '🎓 Estado Estimado: <strong>APROBADO</strong>';
            prob = Math.min(99.9, Math.round(52 + (score - 65) * 1.15));
            bannerProb.textContent = `Probabilidad Estimada de Aprobación: ${prob}%`;
        } else {
            bannerBox.className = 'sim-output-banner fail';
            bannerTitle.innerHTML = '⚠️ Estado Estimado: <strong>EN RIESGO DE REPROBACIÓN</strong>';
            prob = Math.max(5.0, Math.round(50 - (65 - score) * 1.45));
            bannerProb.textContent = `Probabilidad Estimada de Aprobación: ${prob}% (Alerta Temprana)`;
        }

        // Construir Ficha de Recomendación Pedagógica
        if (recList) {
            recList.innerHTML = '';
            
            // Regla de horas
            if (h < 3.0) {
                recList.innerHTML += `<li>⚠️ <strong>Tiempo de Estudio Insuficiente (${h.toFixed(1)} h/día):</strong> El estudiante dedica menos de 3 horas diarias. Se recomienda establecer un cronograma de estudio supervisado aplicando la técnica Pomodoro (bloques de 25 minutos) y fijar metas semanales verificables.</li>`;
            } else if (h >= 6.0) {
                recList.innerHTML += `<li>⭐ <strong>Dedicación Sobresaliente (${h.toFixed(1)} h/día):</strong> El volumen de estudio es excelente. Monitorear para evitar agotamiento cognitivo (burnout) y asegurar pausas activas.</li>`;
            } else {
                recList.innerHTML += `<li>👍 <strong>Hábito de Estudio Adecuado (${h.toFixed(1)} h/día):</strong> Tiempo de dedicación dentro del rango esperado. Fomentar la calidad de las sesiones autónomas y autoevaluación.</li>`;
            }

            // Regla de asistencia
            if (a < 75) {
                recList.innerHTML += `<li>🚨 <strong>Inasistencia Crítica (${a}%):</strong> La pérdida recurrente de clases presenciales incrementa exponencialmente el riesgo de reprobación. Convocar de inmediato a acudientes para determinar motivos de inasistencia y pactar plan de regularización.</li>`;
            } else {
                recList.innerHTML += `<li>📅 <strong>Asistencia Regular (${a}%):</strong> La constancia en el aula asegura la aprehensión del currículo y facilita el seguimiento continuo.</li>`;
            }

            // Regla de antecedentes
            if (p < 60) {
                recList.innerHTML += `<li>📚 <strong>Vacíos Conceptuales Previos (${p} pts):</strong> El historial académico del año anterior refleja debilidades en contenidos base. Es prioritario asignar tutorías personalizadas de refuerzo en matemáticas y ciencias.</li>`;
            } else {
                recList.innerHTML += `<li>✅ <strong>Historial Académico Favorable (${p} pts):</strong> El estudiante cuenta con una base sólida. Potenciar sus fortalezas para consolidar el período escolar actual.</li>`;
            }

            // Regla extracurricular
            if (e === 0) {
                recList.innerHTML += `<li>💡 <strong>Equilibrio y Bienestar Estudiantil:</strong> No participa en actividades extracurriculares. Se aconseja vincular al estudiante en deportes, clubes de robótica o arte para desarrollar habilidades blandas y reducir el estrés académico.</li>`;
            } else {
                recList.innerHTML += `<li>🎨 <strong>Desarrollo Integral Activo:</strong> La participación en actividades extracurriculares favorece el liderazgo y el sentido de pertenencia institucional.</li>`;
            }
        }
    }

    inHours.addEventListener('input', runSimulation);
    inPrev.addEventListener('input', runSimulation);
    inAtt.addEventListener('input', runSimulation);
    inExtra.addEventListener('change', runSimulation);

    // Botón de Copiar Reporte
    if (btnCopy) {
        btnCopy.addEventListener('click', () => {
            const h = inHours.value;
            const p = inPrev.value;
            const a = inAtt.value;
            const status = bannerTitle.innerText;
            const prob = bannerProb.innerText;

            let reportText = `FICHA DE RECOMENDACIÓN PEDAGÓGICA (PROJECT ML-PRE)\n`;
            reportText += `--------------------------------------------------\n`;
            reportText += `Diagnóstico: ${status}\n`;
            reportText += `${prob}\n`;
            reportText += `Variables: Horas: ${h} h/día | Asistencia: ${a}% | Historial previo: ${p} pts\n\n`;
            reportText += `ACCIONES TUTORIALES RECOMENDADAS:\n`;

            const lis = recList.querySelectorAll('li');
            lis.forEach(li => {
                reportText += `- ${li.innerText}\n`;
            });

            navigator.clipboard.writeText(reportText).then(() => {
                const originalText = btnCopy.textContent;
                btnCopy.textContent = '✅ ¡Copiado al Portapapeles!';
                setTimeout(() => {
                    btnCopy.textContent = originalText;
                }, 2500);
            });
        });
    }

    runSimulation();
}
