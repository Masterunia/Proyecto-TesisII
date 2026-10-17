import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from features import preparar_observacion

s=np.array([[0.,0.,np.pi,0.],[1.2,2.,np.pi-1e-6,-3.],[1.2,2.,-np.pi+1e-6,-3.]],dtype=float)
for modo,d in [('baseline',4),('fs1',4),('fs2',5)]:
 a=preparar_observacion(s,modo);assert a.shape==(3,d) and a.dtype==np.float32
 assert np.isfinite(a).all();print(modo,': OK',a.shape)
fs2=preparar_observacion(s,'fs2')
assert np.linalg.norm(fs2[1]-fs2[2])<1e-4, 'Continuidad trigonométrica falló'
assert np.allclose(fs2[:,2]**2+fs2[:,3]**2,1.,atol=1e-6)
try: preparar_observacion([1,2,3], 'baseline'); raise AssertionError('No rechazó entrada inválida')
except ValueError: pass
print('PRUEBAS FE: 5/5 OK; esto verifica transformaciones, NO rendimiento de entrenamiento')
