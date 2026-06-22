# Contenido de las diapositivas

Cada diapositiva se especifica con su texto visible exacto, el visual requerido,
el diseño recomendado, los elementos que no deben aparecer y el tiempo estimado.

---

## Diapositiva 1 — Reconstrucción del discurso legislativo

**Propósito:** Presentar el proyecto, el equipo y el alcance en los primeros treinta segundos.

**Texto visible:**

```
Reconstrucción del discurso legislativo
a partir de transcripciones parlamentarias oficiales

Un pipeline de PLN validado para reconstruir turnos de habla,
separar ruido estructural y habilitar el análisis de distancia discursiva

Ramón Eppens ·  Federico Saroka  ·  Geronimo Fretes  ·  Felipe Merlo
Procesamiento del Lenguaje Natural  ·  ITBA  ·  20/06/2026
```

**Visual:** Tipografía limpia. Sin gráficos. Fondo blanco o institucional.

**Diseño:** Título centrado, subtítulo en peso menor, nombres del equipo abajo.

**No debe aparecer:** URLs, conteos de pruebas, afirmaciones cuantitativas.

**Tiempo estimado:** 0:30

---

## Diapositiva 2 — Objetivo de investigación

**Propósito:** Formular la pregunta de investigación y explicar por qué se necesita un
pipeline propio antes de cualquier análisis.

**Texto visible:**

```
Pregunta de investigación

¿Cómo evolucionó la distancia discursiva entre los bloques
oficialista y opositor en la Cámara de Diputados de la Nación?

Tres problemas previos que resolver

1. El análisis requiere texto atribuido a oradores individuales,
   no el texto crudo de los PDFs.
2. Las transcripciones oficiales contienen ruido estructural que
   corrompe la atribución si no se elimina.
3. La alineación política es temporal: no puede leerse de una
   etiqueta partidaria.

Nota: la métrica es distancia discursiva o semántica —
no es una medición directa de polarización ideológica.
```

**Visual:** Opcional: dos columnas — pregunta a la izquierda, problemas a la derecha.

**No debe aparecer:** Resultados finales, cifras de alineación, términos de polarización
más allá de la aclaración explícita.

**Tiempo estimado:** 1:15

---

## Diapositiva 3 — Por qué los PDFs parlamentarios son difíciles

**Propósito:** Mostrar al público qué significa en la práctica trabajar con el Diario
de Sesiones desde una perspectiva de PLN.

**Texto visible:**

```
Fuente: Diario de Sesiones de la H. Cámara de Diputados de la Nación
Universo candidato del proyecto: sesiones desde 2008

Dificultades del documento
  ·  Diseños de página que cambian entre períodos e impresoras
  ·  Extracción multicolumna con orden de lectura incorrecto
  ·  Encabezados, apéndices e insertos documentales
       mezclados con el discurso
  ·  Marcadores de orador a mitad de bloque o
       divididos entre renglones
  ·  Bloques procedimentales y registros de votación
       intercalados entre turnos
  ·  Acotaciones escénicas y notas editoriales
       incrustadas en el texto de cada orador
  ·  Continuaciones de turno entre páginas y entre bloques
```

**Visual:** Opcional: captura anotada de una página de sesión que muestre columnas,
un marcador incrustado y una acotación. Si hay restricciones de derechos de autor,
omitir y apoyarse en el texto.

**No debe aparecer:** Estadísticas finales del corpus, año de inicio del análisis.

**Tiempo estimado:** 1:00

---

## Diapositiva 4 — Arquitectura del pipeline

**Propósito:** Dar al público un mapa visual del pipeline completo para que las
diapositivas siguientes encajen en una estructura coherente.

**Texto visible:**

```
Etapas del pipeline

COMPLETADAS ✓                       PLANIFICADAS →

Índice oficial de sesiones          Resolución de identidades
  ↓                                   ↓
Descarga de PDFs                    Asignación temporal de bloques
  ↓                                   ↓
Extracción sensible al diseño       Clasificación oficialismo/oposición
  ↓                                   ↓
Segmentación estructural            Agregación por legislador y sesión
  ↓                                   ↓
Detección de marcadores             Embeddings por documento
de orador explícitos                  ↓
  ↓                                 Distancia coseno entre centroides
Reconstrucción de                   de oficialismo y oposición
turnos de habla                     por sesión
  ↓
Clasificación exacta
del contenido
  ↓
Persistencia determinística
y reanudable

```

**Visual:** Diagrama de dos columnas. Columna izquierda: bloques con borde sólido
(completado, verde). Columna derecha: bloques con borde discontinuo (planificado, gris).
Flecha o separador visual entre columnas.

**No debe aparecer:** Valores de distancia, cifras de cobertura, tasas de resolución.

**Tiempo estimado:** 1:00

---

## Diapositiva 5 — Reconstrucción de turnos y clasificación del contenido

**Propósito:** Explicar la contribución algorítmica central: cómo el pipeline asigna
cada carácter fuente a un orador y a un tipo de contenido funcional.

**Texto visible:**

```
Reconstrucción de turnos de habla
  · Cada marcador explícito Sr./Sra. inicia un nuevo turno
  · El contenido sin marcador se hereda del marcador anterior
  · Las barreras procedimentales reinician la atribución
  · El material sin resolver se preserva como no atribuido — nunca se descarta

Clasificación exacta del contenido (por turno, sin pérdida)
  · Texto discursivo       palabras del legislador para el análisis
  · Inserto documental     texto legislativo o documental incorporado
                           a la transcripción
  · Acotación escénica     eventos de cámara (aplausos, cuartos intermedios)
  · Nota editorial         referencias Véase y notas al pie
  · Texto no atribuido     contenido sin contexto activo de orador

Cada segmento conserva página, bloque y offsets de caracteres exactos.
La procedencia nunca se aproxima.
```

**Visual:** Tabla de dos columnas: izquierda = tipo de contenido, derecha = descripción
breve. No usar fragmentos reales de transcripción.

**No debe aparecer:** Conteos cuantitativos de los pilotos (corresponden a la
diapositiva 7).

**Tiempo estimado:** 1:15

---

## Diapositiva 6 — Diseño de validación

**Propósito:** Explicar cómo se verificó la corrección del pipeline antes de cualquier
análisis.

**Texto visible:**

```
Conjunto de validación
  Seis sesiones oficiales heterogéneas, elegidas para someter a prueba
  distintos diseños y patrones de contenido:

  Ordinaria antigua  ·  Continuación  ·  Remota
  Extensa reciente  ·  Informativa  ·  Ordinaria/especial reciente

Qué se auditó en cada piloto
  ✓ Límites estructurales exactos (inicio y fin de las actuaciones)
  ✓ Conteo de marcadores explícitos auditado manualmente
  ✓ Reconstrucción de turnos (atribución explícita vs. heredada)
  ✓ Todos los caracteres de los segmentos de turno clasificados exactamente
    una vez, sin huecos ni superposiciones
  ✓ Conteos exactos de acotaciones, notas editoriales y palabras de discurso

Nota: un evento lógico puede requerir más de un segmento exacto de fuente
cuando cruza un límite de bloque de extracción.

Umbral de calidad automatizado: pytest  ·  Ruff  ·  mypy
Pruebas automatizadas, Ruff y mypy aprobados.
```

**Visual:** Opcional: lista de verificación con íconos.

**No debe aparecer:** Afirmación de que 6 pilotos son representativos de las 400+
sesiones totales.

**Tiempo estimado:** 1:00

---

## Diapositiva 7 — Resultados cuantitativos

**Propósito:** Reportar los resultados cuantitativos verificados de la auditoría sobre
seis pilotos.

**Texto visible:**

```
6 de 6 pilotos aprobaron la validación exacta

  2.924   marcadores explícitos detectados
  2.939   turnos reconstruidos
    697   acotaciones escénicas
     66   notas editoriales
451.523   palabras de discurso retenidas
```

**Visual principal:** `presentation/assets/markers_vs_turns.png`
Barras agrupadas por piloto: marcadores explícitos vs. turnos reconstruidos.

**El gráfico `non_speech_spans.png` NO aparece en esta diapositiva.**
Está disponible en la Diapositiva de respaldo B (ver al final).

**No debe aparecer:** ambos gráficos en la misma diapositiva. No mostrar todos los
números por piloto en texto diminuto. Mantener el titular legible en proyección.

**Tiempo estimado:** 1:00

---

## Diapositiva 8 — Diseño analítico y estado actual

**Propósito:** Conectar el pipeline validado con el diseño del análisis posterior,
presentar el resultado de persistencia validado, y dejar explícito el estado actual.

**Texto visible:**

```
Cálculo planificado

1. Construir documentos legislador-sesión
2. Generar embeddings por legislador
3. Calcular centroides con igual peso por legislador
4. Medir distancia coseno entre oficialismo y oposición

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Resultado ya validado

┌─────────────────────────────────────────────────┐
│  Persistencia determinística y reanudable       │
│                                                 │
│  6/6 pilotos procesados                         │
│  6/6 salidas reutilizadas                       │
│  Mismos resultados, sin reprocesamiento         │
│                                                 │
│  Validación: versión · tamaño · hashes SHA-256  │
└─────────────────────────────────────────────────┘

  →  Pendiente: identidades · alineación · embeddings · distancia longitudinal
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**Visual:** Lista numerada de 4 pasos. Separador. Recuadro con borde sólido
(resultado validado). Línea de pendientes al pie.

**No debe aparecer:** Ningún valor de distancia, serie temporal, porcentaje de
alineación, tasa de resolución de identidades ni nombre de modelo de embedding.
No afirmar que el corpus histórico completo fue procesado.

**Tiempo estimado:** 1:15

---

## Diapositiva 9 — Limitaciones y mejoras posibles

**Propósito:** Presentar con honestidad lo que el pipeline validado aún no garantiza.

**Texto visible:**

```
Limitaciones actuales

· El conjunto de validación es heterogéneo pero reducido (6 sesiones)
· Las etiquetas de orador son formas normalizadas —
  aún no son identidades legislativas estables
· La alineación política es temporal; la membresía en bloques
  requiere fuentes primarias para cada legislador y cada período
· Pueden persistir anomalías de OCR o extracción fuera del piloto
· La distancia semántica final dependerá del modelo de embedding elegido
· No es posible ninguna interpretación causal a partir de la serie
· Distancia semántica ≠ polarización ideológica

Mejoras planificadas

· Ampliar la validación manual a una muestra más sistemática
· Resolver etiquetas observadas a identidades legislativas estables
· Poblar la membresía temporal en bloques desde fuentes oficiales
· Comparar modelos de embedding y reportar sensibilidad
· Cuantificar la incertidumbre y realizar verificaciones de robustez
```

**Visual:** Dos columnas: Limitaciones a la izquierda, Mejoras a la derecha.

**No debe aparecer:** "Vamos a mostrar…" ni ninguna afirmación de resultado futuro.

**Tiempo estimado:** 1:00

---

## Diapositiva 10 — Conclusiones

**Propósito:** Cerrar con tres afirmaciones sólidas y defendibles.

**Texto visible:**

```
Conclusiones

1.  Los PDFs legislativos oficiales requieren PLN consciente
    del diseño del documento antes de cualquier análisis sustantivo.
    La extracción ingenua corrompe silenciosamente la atribución.

2.  El pipeline reconstruye texto a nivel de orador con procedencia
    exacta, clasificación exacta del contenido, cobertura sin pérdida
    y persistencia determinística y reanudable — todo verificado sobre seis

3.  La arquitectura resultante provee una base defendible para el
    análisis de distancia discursiva oficialismo-oposición:
    las decisiones analíticas son transparentes, documentadas
    e independientemente auditables.

Repositorio: https://github.com/GeronimoFretes/argentine-deputies-discursive-distance
```

**Visual:** Conclusiones numeradas en fuente grande y legible. URL del repositorio al pie.

**No debe aparecer:** Resultados de distancia, afirmaciones ideológicas, predicciones
no sustentadas, URL de diapositivas (se incluye en el correo de entrega).

**Tiempo estimado:** 0:45

---

## Diapositiva de respaldo B — Contenido no discursivo por piloto

**Propósito:** Disponible para responder preguntas durante la defensa oral.
No aparece en la secuencia principal. No se numera entre las diez diapositivas.

**Texto visible:**

```
Contenido no discursivo por sesión piloto
(escalas independientes)
```

**Visual:** `presentation/assets/non_speech_spans.png`
Dos paneles independientes: acotaciones escénicas (superior) y notas editoriales (inferior).

**Cuando mostrar:** Solo si el jurado pregunta sobre los conteos de contenido no discursivo
por piloto.

**No debe aparecer** en la secuencia principal.
