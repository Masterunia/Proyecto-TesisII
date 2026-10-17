# Feature Engineering — Tesis cart-pole DDPG

**Avance reproducible (16/10/2026).** Este repositorio documenta el tratamiento de las observaciones en el entrenamiento DDPG P4, y añade dos representaciones candidatas sin atribuirles resultados de entrenamiento no ejecutados.

## Evidencia comprobada
- Modelo base: P4-60K entrenado sobre `s=[x,v,theta,omega]` (float32; theta normalizado a [-pi,pi)).
- Estabilidad nominal media P4-60K: 107.832 s de ventana continua estricta (cinco semillas); `resultados/resumen_nominal_6_modelos.csv`.
- P4-60K es una selección por evaluación de checkpoints, **no** una ablación de Feature Engineering.

## Implementación FE
- **A Baseline**: cuatro estados originales, angle-wrap.
- **B F.S.1**: escalado físico de posición, velocidad, ángulo y velocidad angular. Escalas candidatas `x_max=2.4`, `v_scale=5`, `omega_scale=10`, a definir por el protocolo antes de entrenar.
- **C F.S.2**: mismo escalado físico y reemplazo del ángulo por `sin(theta)` y `cos(theta)`: salida de cinco características).
- Transformaciones en `features/transformaciones.py`, validación en `pruebas/probar_features.py`.

## Ejecución (PowerShell desde esta carpeta)
`python -m pip install numpy`
`python pruebas/probar_features.py`
`python pruebas/resumen_descriptivo.py`

## Interpretación científica
Las pruebas verifican forma, tipo, continuidad angular y dominio de la transformación, pero **NO validan mejoras de control**. Para la ablación pendiente debe entrenarse DDPG desde una condición comparable con observaciones de 4/4/5 entradas, usando mismo entorno, recompensa, hiperparámetros, semillas, presupuesto y evaluación independiente. LQR/ISMC mantienen el estado físico sin transformación de la red.

## Trazabilidad
`entrenamiento_original/` es una copia sin modificaciones de código P4. `resultados/` incluye CSV previos; no son experimentos de las nuevas representaciones.
