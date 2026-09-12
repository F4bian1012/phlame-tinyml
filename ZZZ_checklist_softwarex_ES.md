# Checklist para publicar en SoftwareX — de los ajustes del repo al envío

Framework TinyML-MLOps + banco HIL de medición (Portenta H7). Marca cada casilla en orden.
La regla de oro de SoftwareX: **si los revisores piden cambios extensos DE SOFTWARE, rechazan
el manuscrito** (te invitan a reenviar de cero). Por eso el software debe quedar sólido ANTES
de escribir. Las tres fases van en este orden y no conviene saltarlas.

---

## FASE 0 — Decidir el encuadre (antes de tocar nada)
- [] Elegir el protagonista del paper: **el framework MLOps completo** (→ SoftwareX) o **el banco HIL de medición como instrumento** (→ HardwareX). Recomendado: SoftwareX con el banco HIL como aporte diferenciador dentro del framework.
- [ ] Confirmar que el software es **reutilizable fuera de tu caso** (placas vs motos). SoftwareX valora el potencial de reuso: el texto debe hablar de "clasificación de imágenes en MCU", no solo de tu dataset.
- [ ] Redactar en una frase la contribución: *"framework reproducible de entrenamiento→cuantización→despliegue→medición HIL por fases para CNN en microcontroladores Cortex-M"*.

## FASE 1 — Arreglar el software (bloqueante; hazlo primero)
Los 8 puntos de la revisión del repo, ordenados por prioridad:
- [ ] **(Crítico) Bug de cuantización** en `image_inference.ino`: `(int8_t)pixel-128` desborda para pixel>127. Cambiar a `(int8_t)(pixel - 128)` o `pixel + INT8_MIN`, validando el rango. Es un fallo de corrección numérica: un revisor lo detecta.
- [ ] **(Crítico) Offset -128 hardcodeado**: leer `input->params.zero_point` y `input->params.scale` del tensor en vez de asumir -128. Hace el código genérico para cualquier modelo cuantizado.
- [ ] **(Alto) Op resolvers inconsistentes** entre los `.ino`: unificar el conjunto de operaciones registradas (o documentar por qué difieren por modelo).
- [ ] **(Alto) FOMO con binary-accuracy engañosa**: reportar métrica correcta (F1/precision-recall por celda), no accuracy binaria que infla resultados.
- [ ] **(Medio) `arduino_project.ino` rellena el tensor con ceros** (harness de latencia, no inferencia real): documentarlo explícitamente como "benchmark de latencia" para no inducir a error.
- [ ] **(Medio) Barridos con lr=1e-6 subentrenados** (MobileNetV3Large ~0.5 acc): etiquetarlos como *ablation* o quitarlos de los resultados principales.
- [ ] **(Bajo) Comentario "2MB" vs kTensorArenaSize real = 1MB** (`1024*1024`): corregir el comentario.
- [ ] **(Bajo) `class_names` binario (Class1/Class2) vs docs que mencionan Class3**: alinear código y documentación.
- [ ] Re-ejecutar el pipeline tras los arreglos y **regenerar la matriz de confusión INT8** (antes 96.3%) para que las cifras del paper salgan del código corregido.

## FASE 1B — Aportes científicos a implementar (elevan el paper de "pipeline" a "contribución")
Estos son los aportes que discutimos: convierten un reempaquetado del estado del arte
(entrenar→cuantizar→desplegar, que ya existe) en una contribución publicable. Impleméntalos
como parte de los cambios, porque cambian qué mides y qué muestras.
- [ ] **Reencuadrar el paper alrededor del banco HIL de medición reproducible** como aporte central, NO alrededor del pipeline de compresión (MobileNet/ResNet/SqueezeNet/FOMO cuantizados = estado del arte ya conocido). El diferenciador es el protocolo de medición serial sobre hardware real.
- [ ] **Añadir el eslabón PIL** a la escalera de fidelidad (MIL→SIL→**PIL**→HIL): medir el modelo en el procesador objetivo (o simulador de instrucciones del Cortex-M7) antes del HIL completo, y reportar la brecha entre niveles. Hoy tu banco salta directo a HIL; el eslabón PIL completa la narrativa y da un eje de comparación.
- [ ] **Medición por fases estilo Bartoli et al. (2025)**: instrumentar latencia y energía **desagregadas por fase** (captura/preprocesado → cuantización de entrada → inferencia → post-proceso) usando señales de *trigger* (marcadores serie o GPIO). Esto es lo que casi nadie del nicho hace y es tu ventaja frente a un simple "tiempo total de inferencia".
- [ ] **Comparación cuantitativa con MLPerf Tiny / PICO / CREST**: ejecutar (o mapear) tu banco contra el protocolo de MLPerf Tiny para dar números comparables, y posicionar tu medición por fases frente a lo que reportan PICO (2025) y CREST (2026). Añade una tabla de "qué mide cada framework" (total vs por fases, co-simulación vs medición real, tipo de HW).
- [ ] **Documentar la brecha PC→placa** con datos propios (precisión/latencia en escritorio vs en el Portenta H7), apoyándote en el hallazgo de *Air Learning* (trayectorias ~40% distintas embebido vs escritorio) como evidencia de por qué la medición HIL es necesaria.
- [ ] Reflejar estos aportes en las figuras (máx. 6): escalera MIL→SIL→PIL→HIL, desglose de latencia/energía por fases, y tabla comparativa de frameworks.

## FASE 2 — Dejar el repositorio "publication-ready"
SoftwareX exige repo **GitHub público** (no GitLab) con requisitos concretos:
- [ ] **LICENSE.txt** con licencia open source reconocida por OSI (MIT o Apache-2.0 recomendadas para software; añade la de datos si publicas el dataset).
- [ ] **README.md bien escrito**: propósito, instalación, uso, dependencias con versiones, hardware requerido (Portenta H7 + Vision Shield HM01B0), y un ejemplo mínimo reproducible ("quickstart").
- [ ] **Instrucciones de reproducción de la medición HIL**: cómo conectar la placa, correr `send_multiple_images_serial.py` y `arduino_project_test.ino`, y leer los resultados. Este es tu aporte; que cualquiera lo replique.
- [ ] Fijar **versiones de dependencias** (Python `requirements.txt`/`environment.yml`; versión de Arduino IDE/CLI, core Mbed, TFLite Micro).
- [ ] Estructura de carpetas limpia + `.gitignore` (sin binarios pesados ni datos privados).
- [ ] Añadir **CITATION.cff** y un archivo de ejemplo de datos (o enlace al dataset).
- [ ] **Crear un release versionado** (p. ej. v1.0.0) y **acuñar un DOI con Zenodo** (integración GitHub↔Zenodo). El DOI de código refuerza la reproducibilidad y va en el paper.

## FASE 3 — Preparar el manuscrito con la plantilla oficial
- [ ] Descargar la **plantilla de SoftwareX** (LaTeX `elsarticle` o Word) del Guide for Authors y **no alterar el formato**. Solo se aceptan envíos con la plantilla.
- [ ] Rellenar la **Code Metadata table** obligatoria (versión actual del código, enlace permanente al repo, licencia, requisitos de SO/entorno, lenguajes/herramientas, soporte/contacto).
- [ ] Respetar límites: **máx. 3000 palabras** (cuenta abstract, texto, pies de figura, notas; NO cuenta título, autores, afiliaciones, referencias ni tablas de metadatos) y **máx. 6 figuras**.
- [ ] Estructura típica del artículo de software: *Motivation and significance → Software description (arquitectura, funcionalidades) → Illustrative examples → Impact → Conclusions*.
- [ ] Escribir **Highlights** (3–5 viñetas de ≤85 caracteres).
- [ ] **Abstract** ~150–300 palabras.
- [ ] Reutilizar tus figuras del mapa de literatura y de la medición HIL; entregar **figuras vectoriales** (.pdf/.eps) salvo fotos. Máximo 6 — prioriza: arquitectura del framework, protocolo HIL, resultados de latencia/energía por fases, matriz de confusión.
- [ ] Referencias en **estilo Elsevier numérico (Vancouver)**; citar los 4 trabajos núcleo (CREST, AutoMCU, OASI, Real-time NN 2020), MLPerf Tiny, PICO y Bartoli et al.
- [ ] Añadir sección de **comparación con el estado del arte** (tu banco vs MLPerf Tiny / CREST / AutoMCU) para justificar el aporte.

## FASE 4 — Requisitos administrativos del envío
- [ ] **Declaración CRediT** de contribución de autores.
- [ ] **Data/Code Availability Statement** con el enlace al repo y el DOI de Zenodo.
- [ ] **Cover letter**: qué es el software, por qué encaja en SoftwareX, y su potencial de reuso.
- [ ] Lista de **revisores sugeridos** (nombre, email institucional, afiliación; NO de tu misma institución; que conozcan TinyML/embebido).
- [ ] Verificar situación de la **APC** (SoftwareX es open access; el autor de correspondencia paga si se acepta — confirma si tu institución o financiador lo cubre).
- [ ] Declarar conflictos de interés y confirmar que el trabajo es original y no está en revisión en otra revista.

## FASE 5 — Envío y revisión
- [ ] Enviar por **Editorial Manager** de SoftwareX (no por email).
- [ ] Subir: manuscrito (plantilla), figuras fuente, y el enlace/DOI del repo.
- [ ] Pre-screening editorial → si pasa, **≥2 revisores** (single anonymized).
- [ ] Al recibir revisiones: **solo se aceptan revisiones con ajustes de texto**; si piden cambios grandes de software, tendrás que rehacer el repo y reenviar. Por eso la Fase 1 va primero.
- [ ] Preparar **response letter** punto por punto y resaltar cambios en el manuscrito revisado.
- [ ] Tras aceptación: una copia del código se archiva en el repositorio GitHub de la revista.

---
### Ruta corta si tienes prisa
1. Arreglar los 2 bugs críticos (cuantización + zero_point/scale) → 2. Añadir el aporte mínimo que diferencia el paper: **medición por fases + eslabón PIL + una tabla comparativa con MLPerf Tiny/CREST** → 3. README+LICENSE+DOI Zenodo → 4. Plantilla + Code Metadata table + Highlights (≤3000 palabras, ≤6 figuras) → 5. CRediT + Data Availability + cover letter + revisores → 6. Editorial Manager.
