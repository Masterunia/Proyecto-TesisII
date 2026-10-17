"""Imprime evidencia existente P4; no compara nuevas políticas."""
import csv
from pathlib import Path
p=Path(__file__).resolve().parents[1]/'resultados'/'resumen_nominal_6_modelos.csv'
with p.open(encoding='utf-8-sig',newline='') as f:
    rows=list(csv.DictReader(f))
for x in rows:
 if x['model'].startswith('P4_'):
    print(x['model'],'n=',x['n'],'streak(s)=',x['strict_longest_s'],
          'exitos_10s=',x['successful_10s'],'esfuerzo(N2s)=',x['effort_N2s'])
print('Advertencia: NO son variantes de FE; los modelos difieren también en entrenamiento.')
