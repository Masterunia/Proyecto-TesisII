# Proyecto de Tesis II — Control del péndulo invertido Cart-Pole

**Universidad Nacional de Ingeniería (UNI)**  
**Maestría en Inteligencia Artificial**

## Título de la investigación

**Diseño, simulación y evaluación comparativa de DDPG autónomo y arquitecturas híbridas DDPG–LQR y DDPG–ISMC para el balanceo y la estabilización robusta del péndulo invertido tipo Cart-Pole.**

## Objetivo

Diseñar, simular y comparar tres arquitecturas de control para realizar el balanceo (*swing-up*), la estabilización y la recuperación del sistema Cart-Pole frente a perturbaciones externas.

## Metodología

La investigación se desarrolla mediante simulación computacional utilizando Python, NumPy, Gymnasium y Stable-Baselines3.

Se realizaron diversas campañas experimentales para estudiar el desempeño y la reproducibilidad del algoritmo DDPG, incluyendo:

- Modificación de hiperparámetros y estrategias de exploración.
- Variación de la representación de estados y características de entrada.
- Entrenamiento por currículo y modificaciones de la memoria de experiencias.
- Experimentos de ablación y análisis de sensibilidad.
- Evaluación de TD3 como algoritmo alternativo.
- Implementación y comparación de DDPG–LQR y DDPG–ISMC.

## Principales resultados

Las modificaciones del entrenamiento DDPG produjeron mejoras en determinadas pruebas, pero estas no se confirmaron consistentemente con nuevas semillas independientes.

Posteriormente, se evaluó TD3. En la campaña confirmatoria TD3-03, no se encontró evidencia suficiente de una ventaja reproducible respecto a DDPG.

En la campaña H3, las arquitecturas presentaron los siguientes resultados de estabilidad estricta media:

| Arquitectura | Estabilidad media |
|---|---:|
| DDPG autónomo | 30,19 s |
| DDPG–LQR | 69,65 s |
| DDPG–ISMC | 71,82 s |

Las mejoras observadas en los controladores híbridos se obtuvieron utilizando políticas DDPG previamente entrenadas y sin modificar sus pesos neuronales durante la evaluación.

Estos resultados corresponden a las condiciones simuladas y no constituyen una demostración de estabilidad global.

## Archivos del repositorio

Este repositorio contiene notebooks de Google Colab correspondientes a las campañas de entrenamiento y evaluación de DDPG, experimentos de sensibilidad y ablación, comparaciones DDPG–TD3, pruebas de control híbrido DDPG–LQR y DDPG–ISMC, así como el informe parcial de tesis en PDF.

Los notebooks pueden consultarse directamente desde GitHub seleccionando cada archivo `.ipynb`.

## Alcances

La investigación se limita al modelo dinámico y a los escenarios de simulación evaluados. No incluye implementación sobre un prototipo físico ni certificación formal de estabilidad global.

## Autores

- José Luis Espinoza Chamaya
- Franklin Enrique Alegría Barboza

**Lima, Perú — 2026**
