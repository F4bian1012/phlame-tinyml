# Checklist para publicar en SoftwareX — de los ajustes del repo al envío

Framework TinyML-MLOps + banco HIL de medición (Portenta H7). Marca cada casilla en orden.
La regla de oro de SoftwareX: **si los revisores piden cambios extensos DE SOFTWARE, rechazan
el manuscrito** (te invitan a reenviar de cero). Por eso el software debe quedar sólido ANTES
de escribir. Las tres fases van en este orden y no conviene saltarlas.

> **Estado del repo (branch `Scripts-Depurados`, actualizado):** la **Fase 1 (firmware) está
> CERRADA en el repo** (los 3 bugs críticos — cuantización, zero_point/scale y `delay` — ya
> corregidos y subidos; latencia por fases con DWT->CYCCNT). La Fase 2 (repositorio
> publication-ready) está prácticamente completa (falta release+DOI Zenodo y 2 secciones de
> README). Quedan los aportes científicos (Fase 1B) y el manuscrito (Fases 3–5). Ver marcas
> ✅/⬜ abajo.

---

## FASE 0 — Decidir el encuadre (antes de tocar nada)
- [x] **Protagonista del paper: DECIDIDO → SoftwareX, con PHLAME como framework completo y el banco HIL como aporte diferenciador dentro de él (no como diseño de hardware independiente).**
  - **Por qué NO HardwareX:** el hardware es comercial de terceros (Arduino Portenta H7 + Vision Shield) — no hay PCB, BOM ni diseño físico propio que aportar; HardwareX exige que el lector pueda *construir* un dispositivo a partir del paper, y aquí no hay diseño que construir. El rig de estímulo con ground truth (Fase 1C, camino 2) es el *medio para probar* el software, no el producto.
  - **Por qué SÍ SoftwareX:** el producto se reproduce **clonando un repo** (LICENSE+README+DOI, modelo SoftwareX puro); el aporte diferenciador (escalera MIL→SIL→PIL→HIL con medición por fases) vive en firmware+scripts, es metodología de software; el argumento de reuso (pipeline sobre VWW, no solo el dataset propio) es un argumento de software portable, incoherente con HardwareX.
  - **Consecuencia práctica:** el banco HIL real se redacta como *"la pieza que cierra el lazo de medición dentro del framework"*, coherente con el título validado, la sección Motivation & Significance ya escrita, y la Code Metadata table (Fase 3), que asumen software desde el inicio.
- [ ] Confirmar que el software es **reutilizable fuera de tu caso** (placas vs motos). SoftwareX valora el potencial de reuso: el texto debe hablar de "clasificación de imágenes en MCU", no solo de tu dataset.
- [x] **Dataset de reuso: decisión tomada → Visual Wake Words (VWW), con CIFAR-10 como opción multi-clase.** El demostrador de reuso corre el pipeline completo (train→cuantiza→despliega→MIL/SIL/PIL/HIL) sobre un **benchmark público estándar de TinyML**, no sobre el dataset propio.
  - **Elegido: Visual Wake Words (VWW)** — tarea **binaria** (persona sí/no), encaja con el pipeline Class1/Class2 **sin tocar la cabeza del modelo ni el firmware**; es una de las 4 tareas de **MLPerf Tiny** (comparabilidad directa con la comunidad); admite baja resolución/grises, compatible con la cámara **HM01B0** para la fase HIL real.
  - **Opción secundaria: CIFAR-10** (32×32, 10 clases; tarea de *image classification* de MLPerf Tiny) si se quiere mostrar que el framework escala a multi-clase. Cabe holgado en los 2 MB de flash.
  - **❌ DESCARTADO ImageNet como conjunto de evaluación en el dispositivo:** 1000 clases RGB 224×224 no caben en 2 MB flash / 4 MB arena; desajuste de dominio con el sensor (HM01B0 = 160×120 grises); no es lo que un revisor de TinyML/HIL espera; confunde "reuso del pipeline" con "accuracy en dataset difícil". **ImageNet solo se cita como fuente de *pre-entrenamiento* de los backbones** (MobileNet/ResNet/SqueezeNet fine-tuned) — declararlo explícito en el paper.
- [x] **Frase de contribución — REDACTADA:**
  > *"**PHLAME** es un framework de software abierto y reproducible que cubre todo el ciclo entrenamiento→cuantización INT8→despliegue→medición por fases de clasificadores CNN en microcontroladores ARM Cortex-M, organizando la evaluación en una escalera de cuatro niveles de fidelidad (**MIL→SIL→PIL→HIL**) que hace explícita y medible la brecha entre simulación de escritorio y el dispositivo físico con su cámara en el lazo, y cuyo reuso se demuestra tanto en el caso de aplicación propio como en un benchmark estándar de TinyML (Visual Wake Words)."*
  >
  > **Versión corta (para abstract/highlights):** *"PHLAME evalúa clasificadores TinyML en Cortex-M a través de una escalera de fidelidad MIL→SIL→PIL→HIL con medición de latencia por fases, reproducible y validada en un benchmark estándar (VWW)."*
  - Ensambla las piezas ya decididas: nombre+expansión (PHLAME), la escalera de fidelidad como aporte metodológico central, la medición por fases (DWT), y el argumento de reuso (VWW) que blinda la crítica de "solo tu caso". Consistente con el título validado y con `motivation_significance_PHLAME.md`.
- [x] **Nombre del software: `PHLAME`** → **DECIDIDO.** Expansión: *Phase-Level And Microcontroller Evaluation*. Verificado libre en el nicho TinyML/embebido (sin colisión con MLPerf Tiny, PICO, MLonMCU, U-TOE, RIOT-ML…) y sin colisión de software científico (único homónimo: un grupo de fotónica "PhLAME" en el proyecto TORONE, dominio ajeno — riesgo bajo). Descartado **FLINT** por chocar de lleno con la *Fast Library for Number Theory* (software académico muy citado). **Regla de uso:** escribir siempre el acrónimo expandido en título y abstract — "PHLAME (Phase-Level ... Microcontroller Evaluation)" — para desambiguar.
  - **Título propuesto (validado):** *PHLAME: A Phase-Level Hardware-in-the-Loop Framework for Reproducible Fidelity-Ladder Evaluation of TinyML Image Classifiers on ARM Cortex-M Microcontrollers*.
- [x] **Reservar el nombre** → **HECHO.** (a) Repo GitHub renombrado `Investigacion` → **`phlame-tinyml`** (URL nueva: `github.com/F4bian1012/phlame-tinyml`; GitHub redirige los enlaces viejos automáticamente); (b) **topics añadidos** (10): `tinyml`, `hardware-in-the-loop`, `microcontroller`, `edge-ai`, `arduino-portenta`, `cortex-m7`, `tensorflow-lite-micro`, `model-quantization`, `benchmark`, `phlame`. **Pendiente menor:** actualizar remoto local (`git remote set-url origin https://github.com/F4bian1012/phlame-tinyml.git`). PyPI opcional: solo si empaquetas como librería instalable (SoftwareX no lo exige; el repo clonable basta).
- [x] **Propagar el nombre PHLAME al contenido del repo** → **HECHO.** (a) Título H1 del README → *PHLAME: Phase-Level And Microcontroller Evaluation* ✅; (b) Overview reescrito con la frase de contribución completa y el acrónimo definido ✅; (c) BibTeX del README actualizado al título validado (`@article{phlame_2026, title={PHLAME: A Phase-Level Hardware-in-the-Loop Framework...}}`) ✅; (d) URL `phlame-tinyml` corregida en clone/tree/Citation del README ✅ (se encontró y arregló además una referencia residual al nombre viejo `Investigacion` en las instrucciones `git clone`). **Pendiente menor:** `CITATION.cff` tiene `title: "Phlame: TinyML-MLOps Framework for ARM Cortex-M7"` — ya menciona Phlame pero no coincide textualmente con el título validado de SoftwareX; alinear antes del release/DOI de Zenodo.

## FASE 1 — Arreglar el software (bloqueante; hazlo primero)
Los puntos de la revisión del repo, ordenados por prioridad. Nota: el firmware canónico ahora es
`deployment/hil_firmware/hil_firmware.ino` (renombrado desde `arduino_project_test`/`image_inference`).
- [x] **(Crítico) Bug de cuantización** en `hil_firmware.ino` → **RESUELTO en el branch.** Ahora lee `input->params.scale`/`zero_point` reales del tensor y aplica `q = round(real/scale)+zp` con saturación a `[-128,127]`. *(Nota de campo: el modelo actual espera pixel CRUDO `[0,255]`, no `[0,1]`; normalizar a `[0,1]` colapsaba todo a Class1. El firmware usa `real_value = (float)image_buffer[i]`.)*
- [x] **(Crítico) Offset -128 hardcodeado** → **RESUELTO.** Ya no asume -128; usa `input->params.zero_point`/`scale` del tensor, código genérico para cualquier modelo cuantizado. Imprime `scale`/`zp` en `setup()` para diagnóstico.
- [x] **(Crítico) `delay(1000)` dentro del lazo de inferencia** → **RESUELTO.** Eliminado del `loop()`. La latencia se mide ahora con el contador de ciclos **DWT->CYCCNT** del Cortex-M7 (resolución ~2 ns a 480 MHz), desagregada por fase (PRE/INF/POST/TOTAL) y reportada en líneas `CYC_*`/`US_*`.
- [ ] **(Alto) FOMO con binary-accuracy engañosa**: reportar métrica correcta (F1/precision-recall por celda). *(Nota: FOMO ya no está en este branch; aplica si lo reintroduces.)*
- [ ] **(Medio) Harness de latencia con tensor en ceros**: si mantienes un modo benchmark que rellena el tensor con ceros, documentarlo explícitamente como "benchmark de latencia", no inferencia real.
- [ ] **(Medio) Barridos con lr=1e-6 subentrenados** (MobileNetV3Large ~0.5 acc): etiquetarlos como *ablation* o quitarlos de los resultados principales.
- [x] **(Bajo) ~~Comentario "2MB" vs arena real~~** → **RESUELTO.** El arena ahora es `4 * 1024 * 1024` (**4 MB** en SDRAM) y el comentario del código es coherente.
- [ ] **(Bajo) `class_names` binario (Class1/Class2) vs docs que mencionan Class3**: alinear código y documentación (`models/class_names.txt`).
- [ ] **(Medio) `AllOpsResolver` → `MicroMutableOpResolver`** con solo las ops usadas (menos flash/RAM), o documentar por qué se mantiene el completo. *(Sigue como `AllOpsResolver`.)*
- [ ] **(Pendiente) Re-ejecutar el banco HIL con el firmware ya corregido** y **regenerar la matriz de confusión** (`results/hil/HIL_Confusion_Matrix.png`) + las latencias por fase, para que las cifras del paper salgan del código corregido. **Este es ahora el paso que desbloquea la tabla de resultados.**

## FASE 1B — Aportes científicos a implementar (elevan el paper de "pipeline" a "contribución")
Estos son los aportes que discutimos: convierten un reempaquetado del estado del arte
(entrenar→cuantizar→desplegar, que ya existe) en una contribución publicable. Impleméntalos
como parte de los cambios, porque cambian qué mides y qué muestras.
- [ ] **Reencuadrar el paper alrededor del banco HIL de medición reproducible** como aporte central, NO alrededor del pipeline de compresión (MobileNet/ResNet/SqueezeNet/FOMO cuantizados = estado del arte ya conocido). El diferenciador es el protocolo de medición serial sobre hardware real. *(El README ya documenta el Serial Protocol — buen primer paso.)*
- [x] **Escalera de fidelidad MIL→SIL→PIL identificada en tus scripts reales** → tres eslabones ya implementados: **MIL** = `test_model.py` (float PC), **SIL** = `test_tflite_model.py` (INT8 simulado en PC), **PIL** = `pil_benchmark.py` + firmware serie (procesador real, datos inyectados por serie). *(Ojo taxonómico: lo que llamabas "HIL" es en rigor **PIL** — el chip real ejecuta, pero los estímulos entran como datos digitales, sin sensor físico en el lazo.)* **El eslabón HIL completo se aborda en la Fase 1C (camino 2).**
- [ ] **Medición por fases estilo Bartoli et al. (2025)**: instrumentar latencia y energía **desagregadas por fase** (captura/preprocesado → cuantización de entrada → inferencia → post-proceso) usando *triggers* (marcadores serie o GPIO).
- [ ] **Comparación cuantitativa con MLPerf Tiny / PICO / CREST**: ejecutar (o mapear) tu banco contra el protocolo de MLPerf Tiny; tabla de "qué mide cada framework" (total vs por fases, co-simulación vs medición real, tipo de HW).
- [ ] **Documentar la brecha PC→placa** con datos propios (precisión/latencia en escritorio vs en el Portenta H7), apoyándote en *Air Learning* (~40% de divergencia embebido vs escritorio).
- [ ] Reflejar estos aportes en las figuras (máx. 6): escalera MIL→SIL→PIL→HIL, desglose de latencia/energía por fases, y tabla comparativa de frameworks.

## FASE 1C — Cerrar el lazo HIL REAL (camino 2: cámara física en el lazo)
El banco actual es **PIL** (chip real, datos inyectados por serie). Para reclamar **HIL** con rigor,
el frontend de sensor físico —la cámara **HM01B0** del Vision Shield— debe adquirir la escena real
y el sistema completo correr en tiempo real, con *ground truth* controlado. Esto es trabajo nuevo,
no solo re-etiquetado. Ordenado por dependencia:

**Firmware / software embebido**
- [x] **Modo de captura desde la HM01B0** → **HECHO.** Firmware **dedicado** `deployment/hil_camera_firmware/hil_camera_firmware.ino` (514 líneas): inicializa la HM01B0 (160×120 grises, 30 fps), captura frame real. Resuelto como **sketch separado** (no flag) del firmware PIL canónico (`hil_firmware.ino`), que queda intacto como fallback publicable — cumple el objetivo sin perder el banco PIL.
- [x] **Preprocesado cámara→tensor reproducido** → **HECHO** (misma geometría/cuantización que el pipeline; dominio pixel crudo `[0,255]` ya fijado, consistente con el fix de Fase 1).
- [x] **Fase CAPTURE con DWT->CYCCNT** → **HECHO.** Telemetría completa por fases: `CYC_CAPTURE/PRE/INF/POST/TOTAL` y `US_CAPTURE/PRE/INF/POST/TOTAL`; PRE incluye resize+cuantización; TOTAL = sensor→predicción.
- [x] **Reporte HIL por serie** → **HECHO.** Handshake (`INPUT_SHAPE`, `CAM_INIT:OK/FAIL`, `READY_HIL`) + por inferencia: `TEMP_C`, `TS_MS`, clase, todos los `CYC_*`/`US_*`. Protocolo de trigger simple: `'T'` captura+infiere, `'F1'`/`'F0'` activa/desactiva frame-dump (para validación cruzada SIL sobre el frame real).

**Rig físico y ground truth (el corazón de un HIL creíble)**
- [x] **Montaje de estímulo controlado** → **HECHO, opción (a) elegida.** Monitor mostrando las imágenes del dataset en pantalla completa en secuencia temporizada (`hil_camera_benchmark.py`), con `--settle`/`--gap` configurables.
- [x] **Sincronización estímulo↔etiqueta** → **HECHO.** El host controla la pantalla y dispara el trigger `'T'`; ground truth conocido por construcción (etiqueta del estímulo mostrado).
- [x] **Control de condiciones ambientales** → **HECHO.** Flags `--lux`, `--distance-cm`, `--ambient-temp`, `--notes`; se vuelcan en `results/hil/hil_conditions.json` (el "protocolo ambiental que pide un revisor", según el propio docstring del script).

**Host / script de orquestación**
- [x] **Nuevo script `src/hil_camera_benchmark.py`** → **HECHO** (856 líneas). Orquesta presentación de estímulos, trigger, lectura de predicción+telemetría, emparejado con ground truth.
- [ ] **Generar la matriz de confusión HIL real y el CSV de latencias — EJECUCIÓN PENDIENTE.** El código ya escribe `results/hil/HIL_Confusion_Matrix.png`, `results/hil/hil_latencies.csv` y `hil_conditions.json`, pero `results/hil/` en el repo solo tiene `.gitkeep` — **el banco aún no se ha corrido sobre hardware real.** Esta es la cifra que respalda la palabra "HIL" en el título y el `[TBD]` de `motivation_significance_PHLAME.md`.

**Validación cruzada (el aporte científico que justifica todo)**
- [x] **Mecanismo de validación cruzada implementado** → **HECHO, y más completo de lo pedido.** `--sil-model`/`--mil-model` + `--dump-frames`: recupera el frame exacto capturado por la cámara y lo re-ejecuta en el PC a nivel MIL y SIL, descomponiendo la brecha en 3 componentes atribuibles: **MIL vs SIL** = pérdida por cuantización INT8 (mismo frame); **SIL vs HIL** = divergencia de ejecución chip/PC (TFLM vs TFLite); **PIL vs HIL** = degradación del frontend físico (pantalla/óptica/sensor). Esto es más fino que "cuantificar la brecha PIL→HIL" — la separa en sus causas.
- [ ] **Ejecutar el barrido y obtener las cifras** — pendiente de correr en hardware (bloqueado por lo anterior).
- [ ] Reflejar los **4 niveles completos MIL→SIL→PIL→HIL con datos propios** en la figura de la escalera de fidelidad (Fase 1B) — pendiente de las cifras reales.

## FASE 2 — Dejar el repositorio "publication-ready"
SoftwareX exige repo **GitHub público** (no GitLab) con requisitos concretos:
> **Repo actual:** `github.com/F4bian1012/phlame-tinyml` (branch de trabajo `Scripts-Depurados`; `default_branch = main`). Renombrado y con 10 topics ✅. **Última verificación:** 31 blobs (+2 desde la revisión anterior: `hil_camera_firmware.ino` y `hil_camera_benchmark.py`, Fase 1C ya implementada), estructura limpia (`src/` sin `Por_Depurar/`), 0 refs rotas.
- [x] **LICENSE** con licencia OSI → **HECHO.** MIT presente en la raíz del branch (HTTP 200).
- [x] **README.md bien escrito**: propósito, instalación, uso, dependencias, hardware, quickstart → **HECHO en su mayor parte.** Incluye TOC, Repository Structure, Serial Protocol y Hardware Requirements. *(Faltan la tabla de Results con cifras y la sección Data Availability + Known Limitations — plantillas ya entregadas en `secciones_readme_ES.md`.)*
- [x] **Instrucciones de reproducción de la medición HIL**: el Quickstart documenta `pil_benchmark.py` + `hil_firmware.ino` con el protocolo `#...@`. → **HECHO.** *(Se robustece al añadir la tabla de resultados.)*
- [x] Fijar **versiones de dependencias**: `requirements.txt` (deps Python agrupadas + deps embebidas: arduino-cli, core `arduino:mbed_portenta`, `Chirale_TensorFlowLite`) **+ `installed_packages.txt`** ya subido como entorno canónico exacto. → **HECHO.**
- [x] Estructura de carpetas limpia + `.gitignore`: firmware duplicado eliminado (1 solo `.ino` canónico), cero `model1.h` huérfanos, `results/hil/` creado. → **HECHO.**
- [ ] **(Camino 2) Reorganización de arquitectura para los 4 niveles de fidelidad** — hacer que MIL/SIL/PIL/HIL sean carpetas/artefactos nombrables y separables (ver el bloque de cambios de arquitectura al final del checklist). El firmware pasa a tener **dos modos** (PIL por serie + HIL por cámara); `results/` se separa en `results/pil/` y `results/hil/`; nace `src/hil_camera_benchmark.py`. Documentar los 4 niveles explícitamente en el README.
- [x] Añadir **CITATION.cff** → **HECHO.** Presente en la raíz (HTTP 200). *(Completar nombre de pila y ORCID.)*
- [ ] Añadir un **archivo de ejemplo de datos** o enlace al dataset (sección Data Availability).
- [ ] **Crear un release versionado** (v1.0.0) y **acuñar un DOI con Zenodo** (integración GitHub↔Zenodo). El DOI de código va en el paper. **Pendiente.**

## FASE 3 — Preparar el manuscrito con la plantilla oficial

> ### 📌 Notas de posicionamiento — ¿por qué la comunidad citaría PHLAME?
> *(Argumentario para redactar "Motivation & Significance" e "Impact". Borrador de sección ya escrito en `motivation_significance_PHLAME.md`.)*
> **Razones fuertes (activos de citación):**
> 1. **Timing / nicho naciente:** TinyML∩HIL = 11 papers (2,2 %), casi todos 2025-2026 con ≈0 citas. Los primeros que definen vocabulario + dan herramienta se vuelven la cita fundacional obligada cuando el subcampo madure (2027-2028).
> 2. **Herramienta reusable > resultado:** un software se cita cada vez que se usa/extiende/menciona como *la forma de hacer X*; vida de citación más larga y estable que un paper de resultados. (Ventaja de publicar en SoftwareX.)
> 3. **Vocabulario citable:** la escalera MIL→SIL→PIL→HIL da a la comunidad una forma limpia de nombrar/distinguir niveles que hoy se mezclan (simulado PC vs chip con datos inyectados vs lazo con cámara). La distinción PIL vs HIL verdadero es un gancho de cita por sí solo.
> 4. **Dato de reproducibilidad:** documentar la brecha PC→dispositivo con cifras propias (ref. ~40 % Air Learning; medición por fases Bartoli 2025) da un número concreto que otros citarán para justificar medir en hardware real.
>
> **Razones débiles (a blindar — si no, no citan):**
> 1. *"Solo tu caso"* → mortal. Se neutraliza con **VWW/CIFAR-10** (reuso demostrado en benchmark estándar, no afirmado).
> 2. *No comparable* → anclar a **MLPerf Tiny / CREST / PICO** para entrar en tablas de "related work".
> 3. *Software que no corre* → LICENSE + README quickstart + DOI Zenodo + deps fijadas = infraestructura de citación (ya casi cerrado en Fase 2).
>
> **Síntesis en 1 frase:** *llega temprano a un nicho en formación con una herramienta reusable y un vocabulario limpio (escalera de fidelidad) que otros necesitan para encuadrar su trabajo — siempre que se demuestre reuso en benchmark estándar y el software corra sin fricción.* Novedad = visibilidad; reusabilidad = durabilidad; son inseparables.

- [ ] Descargar la **plantilla de SoftwareX** (LaTeX `elsarticle` o Word) del Guide for Authors y **no alterar el formato**.
- [x] **Borrador de "Motivation & Significance" escrito** → `motivation_significance_PHLAME.md` (~640 palabras, con Highlights y citas marcadas [CIT]). *(Pendiente: insertar 1 cifra propia de brecha tras re-correr el banco.)*
- [ ] Rellenar la **Code Metadata table** obligatoria (versión, enlace permanente al repo, licencia, requisitos de SO/entorno, lenguajes/herramientas, soporte/contacto).
- [ ] Respetar límites: **máx. 3000 palabras** y **máx. 6 figuras**.
- [ ] Estructura típica: *Motivation and significance → Software description → Illustrative examples → Impact → Conclusions*.
- [ ] Escribir **Highlights** (3–5 viñetas de ≤85 caracteres).
- [ ] **Abstract** ~150–300 palabras.
- [ ] Entregar **figuras vectoriales** (.pdf/.eps) salvo fotos. Máximo 6 — prioriza: arquitectura del framework, protocolo HIL, latencia/energía por fases, matriz de confusión.
- [ ] Referencias en **estilo Elsevier numérico (Vancouver)**; citar los 4 núcleo (CREST, AutoMCU, OASI, Real-time NN 2020), MLPerf Tiny, PICO y Bartoli et al.
- [ ] Añadir sección de **comparación con el estado del arte** (tu banco vs MLPerf Tiny / CREST / AutoMCU).

## FASE 4 — Requisitos administrativos del envío
- [ ] **Declaración CRediT** de contribución de autores.
- [ ] **Data/Code Availability Statement** con el enlace al repo y el DOI de Zenodo.
- [ ] **Cover letter**: qué es el software, por qué encaja en SoftwareX, y su potencial de reuso.
- [ ] Lista de **revisores sugeridos** (nombre, email institucional, afiliación; NO de tu misma institución; que conozcan TinyML/embebido).
- [ ] Verificar situación de la **APC** (SoftwareX es open access; confirma si tu institución o financiador la cubre).
- [ ] Declarar conflictos de interés y confirmar originalidad.

## FASE 5 — Envío y revisión
- [ ] Enviar por **Editorial Manager** de SoftwareX (no por email).
- [ ] Subir: manuscrito (plantilla), figuras fuente, y el enlace/DOI del repo.
- [ ] Pre-screening editorial → si pasa, **≥2 revisores** (single anonymized).
- [ ] Al recibir revisiones: **solo se aceptan revisiones con ajustes de texto**; si piden cambios grandes de software, tendrás que rehacer el repo y reenviar. Por eso la Fase 1 va primero.
- [ ] Preparar **response letter** punto por punto y resaltar cambios en el manuscrito revisado.
- [ ] Tras aceptación: una copia del código se archiva en el repositorio GitHub de la revista.

---
## CAMBIOS DE ARQUITECTURA DEL REPO (camino 2 — cámara física en el lazo)
> **✅ IMPLEMENTADO en el repo** (branch `Scripts-Depurados`, repo ahora `phlame-tinyml`). El diseño real difiere del propuesto originalmente en un punto deliberado y mejor: **firmware SEPARADO por sketch**, no dual-mode con conmutador — decisión correcta, ver nota más abajo.

Estructura **real** verificada en el repo:
```
phlame-tinyml/
├── src/
│   ├── test_model.py              # MIL  (float, PC)                    ✅
│   ├── test_tflite_model.py       # SIL  (INT8 simulado, PC)           ✅
│   ├── pil_benchmark.py           # PIL  (chip real, img por serie)    ✅
│   └── hil_camera_benchmark.py    # HIL  (chip real, cámara HM01B0)    ✅ NUEVO
│
├── deployment/
│   ├── hil_firmware/
│   │   ├── hil_firmware.ino       # firmware PIL — intacto, sin tocar  ✅
│   │   └── model.h
│   └── hil_camera_firmware/
│       └── hil_camera_firmware.ino # firmware HIL dedicado (cámara)    ✅ NUEVO (514 líneas)
│
└── results/
    └── hil/                       # HIL_Confusion_Matrix.png, hil_latencies.csv,
                                    # hil_conditions.json — todo implementado,
                                    # AÚN SIN EJECUTAR (solo .gitkeep en el repo)  🟡
```

**Cambios que SÍ se hicieron (verificados):**
1. ✅ **Firmware SEPARADO, no dual-mode** — `hil_camera_firmware.ino` es un sketch propio e independiente; `hil_firmware.ino` (PIL) queda intacto como fallback publicable. *Nota: es una resolución distinta a la propuesta original de "un firmware con conmutador M0/M1", y es la **mejor decisión de las dos**: cero riesgo de romper el banco PIL que ya funciona, cero lógica condicional compartida que pueda introducir un bug cruzado entre modos.*
2. ✅ **Fase CAPTURE con DWT->CYCCNT** — implementada en el firmware HIL dedicado (`CYC_CAPTURE`/`US_CAPTURE`, además de PRE/INF/POST/TOTAL).
3. ✅ **Script `hil_camera_benchmark.py`** — orquesta el rig, dispara captura, empareja predicción↔ground truth, escribe salidas.
4. 🟡 **`results/pil/` vs `results/hil/` — NO se separó como se propuso.** Ambos bancos (PIL y HIL) escriben a `results/hil/` (el nombre de carpeta quedó fijo desde el banco PIL original). **Riesgo:** correr ambos bancos sobrescribe archivos del otro si usan el mismo nombre de salida. Revisar `--output-dir` en ambos scripts antes de correr los dos flujos, o renombrar manualmente entre corridas (`results/pil_*` vs `results/hil_*`) para no perder ninguna cifra.
5. ✅ **Preprocesado consistente** — dominio pixel crudo `[0,255]` documentado y replicado en el firmware HIL.
6. 🟡 **README: falta la tabla explícita de la escalera de fidelidad** (MIL/SIL/PIL/HIL → script → dónde corre → origen de los datos). El README ya documenta cada script y el bench de cámara (sección 5 nueva), pero no como tabla comparativa de un vistazo — pendiente para Fase 1B/3.
7. ⬜ **`.gitignore` de frames capturados**: `hil_camera_benchmark.py` soporta `--dump-frames` a `data/hil_frames/` — verificar que esa carpeta esté en `.gitignore` (no confirmado en el árbol del repo).
8. ✅ **Validación cruzada MIL/SIL/HIL sobre el mismo frame** (`--sil-model`/`--mil-model`) — implementada y **más completa** que lo pedido: descompone la brecha en 3 componentes atribuibles (cuantización, ejecución chip/PC, frontend físico), no solo una cifra PIL→HIL agregada.

> **Principio rector (se cumplió):** el banco PIL no se rompió — quedó intacto como fallback publicable (MIL→SIL→PIL). El banco HIL se añadió *al lado*, no encima, y la comparación PIL-vs-HIL (más MIL/SIL vía frame dump) es precisamente el resultado científico que legitima el eslabón HIL.

---
### Progreso actual (resumen)
- ✅ **Fase 1 (firmware) CERRADA en el repo:** cuantización con `scale`/`zero_point` reales (pixel crudo `[0,255]`), `delay(1000)` eliminado, latencia por fases con DWT->CYCCNT.
- ✅ **Fase 2 (repo) casi cerrada:** LICENSE, CITATION.cff, README completo (título PHLAME, secciones PIL+HIL cámara, Quickstart de 8 pasos), `requirements.txt` + `installed_packages.txt`, estructura limpia, repo renombrado a `phlame-tinyml` con 10 topics. Falta solo el **release + DOI Zenodo** y 2 secciones de README (Results con cifras, Data/Limitations).
- ⬜ **Fase 1B (aportes científicos):** escalera MIL→SIL→PIL→HIL ya identificada e instrumentada en 4 scripts reales; falta **medir y comparar** (ejecución en hardware).
- 🟡 **Fase 1C (HIL real — CAMINO 2):** **arquitectura completa implementada** (firmware dedicado + rig + orquestación + validación cruzada de 3 vías). Falta únicamente **ejecutar el banco sobre hardware físico** para producir `HIL_Confusion_Matrix.png` + `hil_latencies.csv` + `hil_conditions.json` con datos reales — es la única pieza que falta para legitimar la palabra "HIL" del título con una cifra.
- ⬜ **Fases 3–5 (manuscrito y envío):** no iniciadas, salvo el borrador de Motivation & Significance ya escrito.

### Ruta corta si tienes prisa
1. ~~Arreglar los bugs de firmware~~ ✅ → 2. ~~Cerrar la arquitectura HIL real (Fase 1C)~~ ✅ **hecho** → 3. **EJECUTAR el banco HIL sobre hardware** (correr `hil_camera_benchmark.py` con el rig físico montado), medir la brecha PIL→HIL y la descomposición MIL/SIL/HIL, generar las 3 salidas de `results/hil/` → 4. Tabla comparativa con MLPerf Tiny/CREST → 5. Release + DOI Zenodo → 6. Plantilla + Code Metadata table + Highlights (≤3000 palabras, ≤6 figuras) → 7. CRediT + Data Availability + cover letter + revisores → 8. Editorial Manager.
