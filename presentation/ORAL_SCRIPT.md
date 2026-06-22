# Guión oral de la presentación

---

> **CONFIGURACIÓN PENDIENTE:**
> Reemplazar los roles con los nombres reales antes del ensayo final.
> **Elegir UNA distribución definitiva y eliminar la alternativa que no aplique.**
>
> **Si son dos presentadores:**
>
> | Presentador | Diapositivas | Tiempo estimado |
> |---|---|---|
> | Presentador 1 | 1–5 | ~5 min 45 s |
> | Presentador 2 | 6–10 | ~5 min 45 s |
>
> **Si son cuatro presentadores:**
>
> | Presentador | Diapositivas | Tiempo estimado |
> |---|---|---|
> | Presentador 1 | 1–2 | ~1 min 45 s |
> | Presentador 2 | 3–5 | ~2 min 55 s |
> | Presentador 3 | 6–7 | ~1 min 55 s |
> | Presentador 4 | 8–10 | ~2 min 45 s |
>
> Diferencia máxima entre bloques: ~1 min 10 s. Reordenar si se prefiere mayor paridad.

---

Duración total estimada: 11 minutos 55 segundos.

---

## [PRESENTADOR 1]

---

### Diapositiva 1 — Reconstrucción del discurso legislativo (0:30)

Buenas [tardes / noches]. Muchas gracias por estar acá.

Este proyecto trata sobre cómo reconstruir el discurso legislativo a partir de las
transcripciones parlamentarias oficiales: más concretamente, cómo construir el corpus
de texto confiable que hace falta antes de que cualquier análisis sustantivo sea posible.

---

### Diapositiva 2 — Objetivo de investigación (1:10)

La pregunta de investigación que motiva este trabajo es: ¿cómo evolucionó la distancia
discursiva entre los bloques oficialista y opositor en la Cámara de Diputados de la Nación
a lo largo del tiempo?

Para responderla hace falta texto atribuido correctamente a legisladores individuales,
separado del material procedimental y documental, y vinculado a la alineación política
real de cada legislador en la fecha de cada sesión.

Tres problemas bloquean cualquier análisis directo. El primero es que se necesita texto
a nivel de orador: los PDFs de sesiones son un flujo continuo donde discursos, registros
de votación, proyectos de ley completos, aplausos y notas al pie se intercalan sin
separación. No se puede alimentar eso directamente a un modelo. El segundo es que ese
mismo flujo contiene ruido estructural que corrompe la atribución si no se identifica:
un inserto documental o una acotación asignados al legislador equivocado modifican la
medición sin ningún aviso, y esa corrupción es silenciosa. El tercero es que la alineación
política es temporal: un bloque que apoyó al gobierno en un período puede haberlo opuesto
en el siguiente, y un legislador puede haber cambiado de bloque a mitad de período. La
alineación real en la fecha de cada sesión no puede leerse de una etiqueta partidaria fija.

Un punto importante de encuadre antes de continuar: la métrica que buscamos es distancia
discursiva o semántica, no una medición directa de polarización ideológica.

---

### Diapositiva 3 — Por qué los PDFs parlamentarios son difíciles (0:55)

El material fuente es el Diario de Sesiones de la H. Cámara de Diputados, publicado por la
propia Cámara. El universo candidato del proyecto son las sesiones desde 2008.

Desde la perspectiva del pipeline, estos documentos presentan varios desafíos. Los diseños
varían entre períodos e impresoras — dos columnas, columna simple, o combinaciones — y un
extractor ingenuo que lea las columnas en orden incorrecto mezcla el texto silenciosamente.
Encabezados, apéndices e insertos documentales se intercalan con el discurso. Los marcadores
de orador pueden quedar divididos entre renglones o bloques. Secciones procedimentales y
registros de votación aparecen intercalados con los discursos, y dentro de cada turno
aparecen acotaciones escénicas: aplausos, cuartos intermedios, referencias Véase.

Todo eso tiene que identificarse y clasificarse antes de que una sola palabra se asigne
a un legislador.

---

### Diapositiva 4 — Arquitectura del pipeline (0:55)

Esta diapositiva muestra el pipeline completo.

La columna izquierda — completada y validada — parte del índice oficial de sesiones y
avanza por la descarga de PDFs, la extracción sensible al diseño y a las columnas, la
segmentación estructural que separa actuaciones de encabezados y apéndices, la detección
de marcadores de orador explícitos, la reconstrucción de turnos de habla, la clasificación
exacta del contenido y la capa de persistencia determinística que evita reprocesar
documentos ya validados.

La columna derecha — planificada — comienza donde terminamos: resolución de etiquetas de
orador a identidades legislativas estables, asignación de alineación política temporal desde
la membresía en bloques, agregación por legislador y sesión, generación de embeddings y
cálculo de la distancia coseno entre centroides por sesión.

El límite entre las dos columnas es real e intencional. Las etapas planificadas están
diseñadas y documentadas, pero ninguna se ejecutó todavía. No saltamos la validación para
llegar antes a la parte interesante del análisis.

---

### Diapositiva 5 — Reconstrucción de turnos y clasificación del contenido (1:10)

Déjenme explicar las dos etapas centrales.

La reconstrucción de turnos parte de marcadores explícitos. Cada vez que la transcripción
imprime "Sr." o "Sra." seguido de un nombre y el separador punto-guión, detectamos el inicio
de un nuevo turno. El turno continúa hasta el próximo marcador. Cuando una barrera
procedimental interviene — un bloque de votación, una sección de trámite — la atribución se
reinicia, porque no hay forma confiable de saber quién habla después sin un marcador nuevo.
El material que aparece antes de cualquier marcador en una sesión se preserva como turno
no atribuido; nunca se descarta ni se asigna silenciosamente.

Cada segmento conserva número de página, índice del bloque fuente y offsets de caracteres
exactos. La procedencia no se aproxima ni se infiere.

Luego, cada turno se clasifica a nivel de segmento exacto. Cada carácter termina en
exactamente una de cinco categorías: texto discursivo, inserto documental, acotación
escénica, nota editorial o texto no atribuido. Solo el texto discursivo entra en el análisis;
los demás se preservan en la salida pero se excluyen del cálculo de distancia. Este diseño
sin pérdida significa que si una clasificación resulta incorrecta, puede corregirse sin
re-correr la extracción: el carácter original sigue ahí con su procedencia.

---

**[Transición a PRESENTADOR 2]**

---

## [PRESENTADOR 2]

---

### Diapositiva 6 — Diseño de validación (0:55)

Gracias. Ahora voy a explicar cómo verificamos que el pipeline efectivamente funciona.

Elegimos seis sesiones oficiales con diversidad estructural deliberada: una ordinaria antigua,
una de continuación, una remota durante la pandemia, una ordinaria larga reciente, una
puramente informativa sin legislación y una que combina puntos ordinarios y especiales. Son
casos de validación elegidos para cubrir variaciones estructurales conocidas, no una muestra
estadísticamente representativa del corpus completo.

Para cada una, realizamos cinco auditorías. Verificamos los límites estructurales exactos —
dónde empiezan y terminan las actuaciones. Comparamos el conteo de marcadores del parser
contra un recuento manual del mismo documento. Verificamos la reconstrucción de turnos.
Confirmamos que todos los caracteres de los segmentos de turno quedan clasificados
exactamente una vez, sin huecos ni superposiciones. Y verificamos que los conteos de
acotaciones escénicas, notas editoriales y palabras de discurso coinciden con los valores
esperados codificados en el conjunto de pruebas.

Las pruebas automatizadas, Ruff y mypy aprobaron.

---

### Diapositiva 7 — Resultados cuantitativos (0:55)

Acá están los números de los seis pilotos.

El parser detectó 2.924 marcadores explícitos y reconstruyó 2.939 turnos. Los quince turnos
adicionales corresponden a contenido preservado como no atribuido: el pipeline crea esos
turnos en lugar de descartar el texto o asignarlo silenciosamente a un orador.

El clasificador encontró 697 acotaciones escénicas y 66 notas editoriales — los eventos
no discursivos que de otro modo contaminarían el análisis — y clasificó 451.523 palabras
como discurso retenido en los seis pilotos.

El gráfico muestra marcadores versus turnos reconstruidos por piloto. Los seis pilotos
aprobaron la validación exacta: todos los caracteres de los segmentos de turno quedaron
clasificados exactamente una vez, sin huecos ni superposiciones. Los conteos de acotaciones
y notas por piloto están disponibles en la diapositiva de respaldo para preguntas.

---

### Diapositiva 8 — Diseño analítico y estado actual (1:05)

El cálculo primario planificado sigue cuatro pasos: construir documentos legislador-sesión
agregando todos los segmentos de texto discursivo elegibles de un legislador en una sesión;
generar un embedding por documento; calcular un centroide por lado con igual peso para cada
legislador — no ponderado por cantidad de palabras — para que ningún orador prolífico domine
la representación de su bloque; y medir la distancia coseno entre los centroides normalizados
de oficialismo y oposición por sesión.

El resultado ya validado es la capa de persistencia determinística y reanudable. En la
primera ejecución, las seis sesiones piloto se procesaron correctamente y todos los conteos
coincidieron con la línea de base validada. En la segunda ejecución, las seis salidas se
reutilizaron sin volver a procesar los documentos. Para decidir si una salida sigue siendo
válida, el pipeline comprueba versión, rutas, tamaños y hashes SHA-256. Esta capa ya
funciona por documento sobre los seis pilotos; el procesamiento histórico completo es la
etapa siguiente.

---

### Diapositiva 9 — Limitaciones y mejoras posibles (0:50)

Voy a ser directo sobre lo que el trabajo actual todavía no garantiza.

El conjunto de validación es estructuralmente diverso, pero son seis sesiones. Documentos
atípicos pueden revelar casos borde todavía no encontrados.

Las etiquetas de orador son formas normalizadas impresas, no identificadores estables de
legisladores. Un apellido compartido por dos diputados en el mismo período no puede
resolverse sin evidencia adicional. La alineación política requiere membresía temporal en
bloques documentada con fuentes primarias; no puede inferirse de etiquetas partidarias.

El resultado final va a ser sensible al modelo de embedding elegido. Planeamos comparar
modelos y reportar esa sensibilidad.

Y el punto de encuadre más importante: una serie de distancia coseno no es una explicación
causal. Solo mide cuán diferente fue el lenguaje de cada lado. Distancia semántica no
equivale a polarización ideológica.

---

### Diapositiva 10 — Conclusiones (0:40)

Tres conclusiones, ninguna dependiente de un resultado aún no generado.

Primera: los PDFs legislativos oficiales no pueden analizarse directamente. La complejidad
estructural del Diario de Sesiones requiere un pipeline de PLN consciente del documento
antes de que cualquier medición sea posible. Construimos y validamos ese pipeline.

Segunda: el pipeline reconstruye texto a nivel de orador con procedencia exacta de fuente,
clasificación exacta del contenido, cobertura sin pérdida y persistencia determinística —
todo verificado sobre seis sesiones estructuralmente heterogéneas.

Tercera: la arquitectura resultante provee una base defendible para el análisis de distancia
discursiva oficialismo-oposición, con decisiones analíticas transparentes, documentadas e
independientemente auditables.

Muchas gracias.

---

*Duración total estimada: 11 minutos 30 segundos a 12 minutos.*
*Presentador 1: diapositivas 1–5 ≈ 5 minutos 40 segundos.*
*Presentador 2: diapositivas 6–10 ≈ 5 minutos 45 segundos (incluye breve transición).*
