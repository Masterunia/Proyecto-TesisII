"""Feature Engineering físico para el entorno CartPole de cuatro estados.
NO modifica los pesos P4-60K ni afirma equivalencia de modelos.
Entradas: [x (m), x_dot (m/s), theta (rad), theta_dot (rad/s)].
"""
import numpy as np

MODOS = ("baseline", "fs1", "fs2")

def preparar_observacion(estado, modo="baseline", x_max=2.4, v_scale=5.0, omega_scale=10.0):
    """Acepta observación 1D o matriz N x 4. La salida es float32."""
    a=np.asarray(estado, dtype=np.float64)
    if a.ndim not in (1,2) or a.shape[-1]!=4 or not np.all(np.isfinite(a)):
        raise ValueError("Se requiere un estado finito de 4 variables")
    if modo not in MODOS: raise ValueError("Modo desconocido")
    if min(x_max,v_scale,omega_scale)<=0: raise ValueError("Escalas positivas obligatorias")
    x,v,theta,omega=(a[...,j] for j in range(4))
    # Mantener exactamente la convención angular del entorno original.
    theta=((theta+np.pi)%(2*np.pi))-np.pi
    if modo=="baseline":
        b=np.stack([x,v,theta,omega],axis=-1)
    elif modo=="fs1":
        b=np.stack([x/x_max,v/v_scale,theta/np.pi,omega/omega_scale],axis=-1)
    else:
        b=np.stack([x/x_max,v/v_scale,np.sin(theta),np.cos(theta),omega/omega_scale],axis=-1)
    return b.astype(np.float32)

def etiquetas_variables(modo):
    return {'baseline':['x','v','theta','omega'],
            'fs1':['x/x_max','v/v_scale','theta/pi','omega/omega_scale'],
            'fs2':['x/x_max','v/v_scale','sin(theta)','cos(theta)','omega/omega_scale']}[modo]
