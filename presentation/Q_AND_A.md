# Preguntas y respuestas anticipadas

Las respuestas son concisas y técnicamente defendibles. Cada afirmación cuantitativa
remite a la tabla de pilotos verificada en `docs/TEAMMATE_HANDOFF.md` o a los módulos
fuente validados. Ninguna respuesta inventa resultados.

---

## P1. ¿Por qué no analizar directamente el texto extraído de los PDFs?

El texto extraído de una transcripción oficial mezcla en un único flujo continuo discursos,
registros de procedimientos parlamentarios, recuentos de votos, textos completos de proyectos
de ley propuestos, aplausos, avisos de cuarto intermedio y referencias cruzadas Véase.
Los diseños multicolumna, si no se manejan, producen un orden de lectura mezclado. Sin
segmentación estructural y clasificación del contenido, un modelo entrenado sobre ese texto
crudo estaría aprendiendo de una mezcla de discurso, procedimiento y texto legislativo
formal, en lugar de lo que los legisladores realmente dijeron. La contaminación sería
silenciosa y variaría en grado de sesión en sesión.

---

## P2. ¿Por qué usar el turno de habla como unidad de reconstrucción?

El turno de habla es la unidad natural de atribución en el discurso parlamentario: una
persona habla hasta que comienza otra. Los párrafos y las páginas son artefactos
tipográficos del proceso de impresión y no se alinean con quién está hablando. Agregar a
nivel del documento legislador-sesión — todos los turnos elegibles del mismo legislador en
una sesión — da una representación textual vinculada a un único orador y a una membresía
política en un bloque único en una fecha única. Esa es la unidad que requiere el cálculo
de distancia.

---

## P3. ¿Cómo se distinguen las acotaciones escénicas del discurso?

Las acotaciones escénicas en las transcripciones parlamentarias argentinas aparecen en dos
formas. La primera es una construcción entre paréntesis: `(Aplausos)`, `(Risas)` o eventos
similares breves de cámara. Estas se detectan contra un vocabulario auditado explícito de
formas permitidas; cualquier paréntesis que no esté en ese vocabulario no se clasifica como
acotación. La segunda forma comienza con un prefijo de guión seguido de una frase de acción
en tiempo presente: `— El señor diputado hace uso de la palabra fuera del micrófono.` Estas
se detectan mediante patrones anclados al inicio de línea que requieren el prefijo de guión
y frases verbales auditadas específicas. Ningún método hace suposiciones: si un span no
coincide con un patrón validado, permanece como texto discursivo.

---

## P4. ¿Por qué se preservan stopwords, gramática y nombres propios?

El pipeline está diseñado para medir distancia discursiva — diferencias en la forma en que
los dos lados usan el lenguaje en su conjunto, incluyendo palabras funcionales, estructuras
gramaticales y patrones retóricos. Eliminar stopwords o aplicar stemming destruiría
exactamente la variación estilística y sintáctica que la medición busca capturar. Solo se
excluyen los artefactos estructurales — acotaciones escénicas, notas editoriales e insertos
documentales — porque corresponden a texto legislativo o documental incorporado a la
transcripción taquigráfica, no a las propias palabras del legislador. El clasificador
identifica ese material en el texto impreso; no establece necesariamente cómo fue entregado
en la sesión.

---

## P5. ¿Por qué agregar el texto por legislador y sesión en lugar de usar los turnos individuales?

Los turnos individuales son muy cortos. Un legislador puede intervenir brevemente varias
veces en una sesión — una pregunta, una cuestión de orden, un comentario procedimental —
además de dar un discurso más extenso. Promediar embeddings de turnos cortos de forma
independiente y luego promediar otra vez entre legisladores daría peso desproporcionado a
los intervenientes frecuentes y breves. Agregar todos los turnos elegibles en un documento
por legislador por sesión y generar el embedding del agregado produce una representación
estable por persona por momento político. Esa es la unidad alrededor de la cual está
diseñado el cálculo de distancia coseno.

---

## P6. ¿Por qué ponderar a cada legislador por igual dentro de su lado, en lugar de hacerlo por cantidad de palabras?

La ponderación por cantidad de palabras permitiría que un número reducido de legisladores
que hablan mucho domine el centroide del lado. Un legislador que habla cuarenta minutos y
uno que da un discurso de cinco minutos tienen el mismo voto en el recinto; la métrica de
distancia debería reflejar la misma paridad. La ponderación igual también hace que la
métrica sea menos sensible a la variación en el estilo de habla entre individuos y más
representativa del bloque en su conjunto.

---

## P7. ¿Por qué utilizar distancia coseno y no, por ejemplo, distancia euclidiana o un clasificador?

La distancia coseno entre embeddings normalizados es la métrica de similitud estándar para
vectores de oraciones y documentos: mide el ángulo entre los dos centroides de lado en el
espacio de embeddings, sin importar las diferencias en magnitud. La distancia euclidiana
confunde dirección con escala. Un clasificador requeriría datos de entrenamiento etiquetados
con etiquetas de alineación de referencia y no generalizaría entre modelos de embedding con
la misma transparencia. La distancia coseno está bien motivada, es agnóstica al modelo
siempre que se use el mismo espacio de embeddings para ambos lados, y es directa de
interpretar y de volver a ejecutar.

---

## P8. ¿Esta medida representa polarización ideológica?

No. La métrica mide distancia semántica o discursiva — qué tan diferente es el lenguaje
usado por los dos lados en un espacio de embeddings dado. La distancia lingüística es un
indicador o correlato de la diferenciación política, pero no es una medición directa de
ideología. No podemos inferir de una distancia mayor que el oficialismo se desplazó más
a la derecha o que la oposición se desplazó más a la izquierda. Solo podemos decir que el
lenguaje de los dos lados fue más o menos similar en una sesión determinada. La
interpretación causal o ideológica requiere evidencia adicional más allá de la métrica.

---

## P9. ¿Qué tan representativos son los seis pilotos de las más de 400 sesiones candidatas?

No son una muestra aleatoria ni estadísticamente representativa. Fueron elegidos
deliberadamente para cubrir tipos de documentos estructuralmente distintos: una sesión
ordinaria antigua, una de continuación, una sesión remota, una sesión ordinaria larga y
reciente, una sesión informativa, y una sesión reciente que combina puntos ordinarios y
especiales. El objetivo era someter al parser a pruebas ante variaciones estructurales
conocidas, no estimar una tasa de cobertura para el corpus completo. Ampliar el conjunto de
validación de forma más sistemática está listado como mejora planificada.

---

## P10. ¿Qué falta para que el análisis longitudinal pueda ejecutarse?

Cuatro cosas. Primero, el pipeline por lotes debe procesar el conjunto completo de sesiones
candidatas y producir un inventario de oradores: una tabla con cada etiqueta de orador
impresa distinta, la fecha de sesión, y los conteos de turnos y palabras. La capa de
persistencia ya validada sobre los seis pilotos está diseñada para escalar a ese procesamiento.
Segundo, cada etiqueta de orador observada debe resolverse a una identidad legislativa estable
mediante la tabla de alias con validez temporal y el protocolo de QA documentado en
`docs/MANUAL_IDENTITY_QA_PROTOCOL.md`. Tercero, cada legislador resuelto debe recibir una
alineación política — núcleo oficialista o núcleo opositor — a través de la tabla de
membresía temporal en bloques documentada en `docs/POLITICAL_METADATA_METHODOLOGY.md`.
Cuarto, debe elegirse el modelo de embedding, generarse los documentos legislador-sesión,
y calcularse las distancias por sesión. Solo entonces existe una serie temporal para
reportar.

---

## P11. ¿Cómo se tratarán los bloques ambiguos o independientes?

Los bloques que no pueden clasificarse de forma confiable como oficialistas u opositores
para un período dado se etiquetan como `ambiguous_independent`. Los bloques que representan
roles no legislativos, administrativos o procedimentales se etiquetan como `excluded`.
Ninguna de las dos categorías entra en el cálculo primario de distancia. Pueden aparecer
en análisis de sensibilidad destinados a verificar cuánto depende el resultado de las
decisiones de frontera. Un bloque nunca se fuerza hacia una categoría central para aumentar
la cobertura analítica. La regla es: ante la duda, afuera.

---

## P12. ¿Cómo se garantiza la reproducibilidad?

El descubrimiento de sesiones es determinista: dado el mismo índice oficial de sesiones de
la Cámara, produce la misma lista de registros. La extracción de PDFs es reanudable y
produce el mismo texto en ejecuciones repetidas para el mismo documento; el conjunto de
pruebas lo verifica. La segmentación estructural y el parsing de turnos son funciones puras
del texto extraído, sin componente aleatorio. Cada segmento generado registra su posición
mediante offsets de caracteres (no de bytes) en el bloque fuente, junto con el número de
página y el índice del bloque; esa procedencia no se aproxima. La capa de persistencia
verifica versión del pipeline, rutas, tamaños y hashes SHA-256 antes de reutilizar cualquier
salida. Los datos de referencia se almacenan en CSVs versionados con procedencia explícita
en cada fila, y un validador determinista verifica la integridad del esquema. Cualquier
resultado del pipeline puede reproducirse desde cero a partir de los mismos PDFs fuente.

---

## P13. ¿Cómo se garantiza que una salida reutilizada siga siendo válida?

Cada manifiesto registra la versión del pipeline, las rutas, los tamaños y los hashes SHA-256
de las fuentes y de las salidas. Una salida solo se reutiliza si todos esos valores coinciden.
Si cambia una fuente, falta un archivo o un hash deja de coincidir, el documento se procesa
nuevamente.

---

## P14. ¿Esto significa que ya procesaron todo el corpus histórico?

No. La capa de persistencia fue validada sobre las seis sesiones piloto. El procesamiento
histórico completo requiere todavía ejecutar el pipeline por lotes sobre el conjunto completo
de sesiones candidatas, resolver identidades y agregar la metadata política temporal.

---

## P15. ¿Por qué hay más turnos reconstruidos que marcadores explícitos?

Porque el pipeline preserva como turnos no atribuidos los segmentos que no tienen un contexto
explícito de orador activo. En los seis pilotos hubo quince turnos de este tipo. No
representan marcadores inventados ni errores de atribución: son contenido genuino del
documento que el pipeline captura en lugar de descartar o asignar silenciosamente a un
orador previo.
