# Lista de verificación final

Completar antes de presentar o entregar. Cada ítem debe ser confirmado por un revisor
humano; este documento no puede auto-aprobarse.

---

## Información de la presentación

- [ ] Nombres completos de los integrantes reemplazados en la diapositiva 1
      (actualmente: [PRESENTADOR 1], [PRESENTADOR 2], [PRESENTADOR 3], [PRESENTADOR 4])
- [ ] Nombre de la materia completado en la diapositiva 1
- [ ] Institución completada en la diapositiva 1
- [ ] Fecha completada en la diapositiva 1
- [ ] URL del repositorio en la diapositiva 10 confirmada:
      `https://github.com/GeronimoFretes/argentine-deputies-discursive-distance`
- [ ] URL de las diapositivas incluida en el correo de entrega
- [ ] Permisos de visualización configurados en modo solo lectura

---

## Asignación de presentadores

- [ ] **Elegir UNA distribución definitiva y eliminar la alternativa antes del ensayo final**
      (las dos tablas en `ORAL_SCRIPT.md` son opciones mutuamente excluyentes)
- [ ] Cantidad real de integrantes confirmada
- [ ] Si son **dos presentadores**: Presentador 1 cubre diapositivas 1–5 (~5 min 45 s),
      Presentador 2 cubre diapositivas 6–10 (~5 min 45 s)
- [ ] Si son **cuatro presentadores**: ver tabla en `ORAL_SCRIPT.md` §CONFIGURACIÓN PENDIENTE
      y redistribuir el guión con los nombres reales antes del ensayo
- [ ] Nombres reales reemplazados en el deck (no dejar placeholders [PRESENTADOR N])
- [ ] Transición entre presentadores ensayada

---

## Contenido y afirmaciones

- [ ] Todo el texto visible en las diapositivas está en español
- [ ] Todo el texto del guión está en español
- [ ] Todas las preguntas y respuestas están en español
- [ ] Cada afirmación cuantitativa en las diapositivas remite a una fuente verificada:
      - Conteos de pilotos (2.924 marcadores, 2.939 turnos, 697 acotaciones,
        66 notas, 451.523 palabras) → tabla de pilotos verificada en
        `docs/TEAMMATE_HANDOFF.md`
      - El resultado de persistencia (6/6 en dos ejecuciones) → ejecuciones
        documentadas del pipeline de persistencia validado
- [ ] No aparece ningún resultado final no validado:
      sin serie de distancia, sin porcentaje de cobertura de alineación, sin tasa de
      resolución de identidades, sin año de inicio del análisis, sin conteo del corpus
      histórico completo
- [ ] La frase "polarización ideológica" no aparece como descripción de la métrica;
      se usa "distancia discursiva" o "distancia semántica"
- [ ] Las diapositivas no muestran el conteo numérico de pruebas automatizadas;
      solo se dice "las pruebas automatizadas, Ruff y mypy aprobaron"
- [ ] Los valores 412 y 398 no aparecen en ningún texto visible de las diapositivas
      ni en el guión
- [ ] Los offsets están descritos como offsets de caracteres (no de bytes) en slides y guión
- [ ] Los quince turnos adicionales están descritos como turnos no atribuidos,
      no como errores de atribución ni marcadores inventados
- [ ] La cobertura exacta está acotada a los segmentos de turno asignados,
      no afirmada para todo el PDF incluyendo encabezados y apéndices
- [ ] Los insertos documentales no están descritos como "leídos en voz alta";
      se usan las frases "texto legislativo o documental incorporado a la transcripción"
      o equivalentes
- [ ] El cuerpo hablado (desde ### Diapositiva 1) tiene entre 1.450 y 1.600 palabras
      (verificado con `python -c "..."` o equivalente)

---

## Diapositiva 7

- [ ] Solo `markers_vs_turns.png` aparece en la diapositiva 7 principal
- [ ] `non_speech_spans.png` está en la Diapositiva de respaldo B, no en la secuencia
      principal
- [ ] El titular de la diapositiva 7 es "6 de 6 pilotos aprobaron la validación exacta"
      (o equivalente)
- [ ] Las cinco tarjetas métricas (2.924, 2.939, 697, 66, 451.523) son legibles en
      proyección sin superponerse con el gráfico

## Diapositiva 8

- [ ] La diapositiva 8 muestra exactamente cuatro pasos analíticos visibles
      (no seis)
- [ ] La diapositiva 8 incluye el recuadro de persistencia validada
      (6/6 procesados, 6/6 reutilizados, validación por versión y hashes SHA-256)
- [ ] La línea de pendientes cierra el cuadro: identidades · alineación · embeddings ·
      distancia longitudinal
- [ ] No queda ningún placeholder de "resultado adicional opcional" en la diapositiva 8

---

## Estructura de las diapositivas

- [ ] Conteo de diapositivas: **10 diapositivas** (título hasta conclusiones)
- [ ] La diapositiva 4 distingue claramente etapas completadas (borde sólido / verde)
      de planificadas (borde discontinuo / gris) en el diagrama del pipeline
- [ ] La diapositiva 6 afirma explícitamente que los seis pilotos son heterogéneos
      pero no una muestra estadísticamente representativa
- [ ] La diapositiva 7 principal incluye únicamente markers_vs_turns.png
non_speech_spans.png está disponible como diapositiva de respaldo
- [ ] La diapositiva 10 tiene la URL del repositorio
- [ ] **No hay placeholder de URL de diapositivas** en la diapositiva 10
      (la URL se incluye en el correo de entrega, no en el deck)

---

## Duración

- [ ] Duración total de habla estimada: **12 minutos 30 segundos** (máximo: 13 minutos)
- [ ] Duración del ensayo completo registrada: _______ minutos
- [ ] Si el espacio incluye defensa oral, reservar tiempo adicional para preguntas

---

## Gráficos

- [ ] `presentation/assets/markers_vs_turns.png` existe y fue generado por
      `uv run python presentation/generate_charts.py`
- [ ] `presentation/assets/non_speech_spans.png` existe y fue generado por
      el mismo comando
- [ ] Ambos gráficos miden exactamente **1600 × 900 píxeles**
- [ ] Ambos gráficos son legibles en pantalla completa en la resolución de proyección
- [ ] El gráfico de marcadores vs. turnos muestra barras agrupadas con valores sobre
      cada barra y leyenda en español
- [ ] El gráfico de contenido no discursivo tiene dos paneles con escalas independientes
      y títulos en español
- [ ] Las etiquetas de piloto en los gráficos están en español
- [ ] Los caracteres con tilde/acento renderizan correctamente (sin glifos vacíos)
- [ ] El conteo de palabras de discurso no aparece mezclado con los conteos de spans
- [ ] Los gráficos están incrustados en las diapositivas con resolución suficiente

---

## Repositorio y reproducibilidad

- [ ] `uv run pytest` pasa en la rama que se presenta
- [ ] `uv run ruff check .` pasa
- [ ] `uv run mypy` pasa
- [ ] `uv run python presentation/generate_charts.py` genera ambos PNGs sin errores

---

## Ensayo final

- [ ] Ensayo completo realizado con todos los presentadores
- [ ] Transición entre presentadores practicada
- [ ] Respuestas a preguntas revisadas con `presentation/Q_AND_A.md`
      (incluye P13–P15: persistencia válida, corpus histórico incompleto,
      turnos no atribuidos)
- [ ] Notas del presentador agregadas a las diapositivas si es necesario
- [ ] Diapositivas probadas en el equipo de presentación o en la plataforma
      de videoconferencia a utilizar

---

## Ítems que requieren verificación humana antes de presentar

1. **Nombres de los integrantes**: reemplazar todos los placeholders [PRESENTADOR N].
2. **Materia e institución**: completar en la diapositiva 1.
3. **Fecha**: actualizar en la diapositiva 1.
4. **URL de las diapositivas**: completar una vez que el deck esté subido.
5. **Asignación de presentadores**: confirmar cantidad de integrantes y redistribuir
   el guión según la tabla en `ORAL_SCRIPT.md`.
6. **Resultado de persistencia**: confirmar que los valores 6/6 (primera ejecución)
   y 6/6 (segunda ejecución) siguen siendo actuales en la rama que se presenta.
7. **Distribución de presentadores**: elegir UNA distribución definitiva y eliminar la
   alternativa del guión antes del ensayo final.
8. **URL de diapositivas**: completar en el correo de entrega, no en el deck.
