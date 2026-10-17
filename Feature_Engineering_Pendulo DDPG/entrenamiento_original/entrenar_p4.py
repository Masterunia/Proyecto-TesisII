"""P4: fine-tuning DDPG puro desde P2-40K, recompensa local y resets mixtos."""
from __future__ import annotations
import argparse, hashlib, json, math
from datetime import datetime
from pathlib import Path
import numpy as np
import gymnasium as gym
from stable_baselines3 import DDPG
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv
from envs.cartpole_continuous_env import CartPoleContinuousEnv


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()


class P4Env(gym.Wrapper):
    def __init__(self, env, seed, upright_probability=.30, local_weight=1.0):
        super().__init__(env)
        self.rng=np.random.default_rng(seed)
        self.upright_probability=upright_probability
        self.local_weight=local_weight
        self.n_capture=0; self.got_capture=False

    def reset(self, **kwargs):
        obs, info=self.env.reset(**kwargs)
        self.n_capture=0; self.got_capture=False
        # Formación de dos habilidades: swing-up desde abajo y corrección local.
        # Modificar el estado tras reset es equivalente a establecer condiciones iniciales;
        # las ecuaciones, límites y espacio de acción permanecen intactos.
        if self.rng.random()<self.upright_probability:
            x=self.rng.uniform(-.55,.55)
            xd=self.rng.uniform(-.25,.25)
            theta=self.rng.uniform(-.16,.16)
            omega=self.rng.uniform(-.35,.35)
            self.unwrapped.state=np.array([x,xd,theta,omega],dtype=np.float32)
            obs=self.unwrapped._get_obs()
            info=dict(info);info['reset_mode']='upright'
        else:
            info=dict(info);info['reset_mode']='downward'
        return obs,info

    def step(self, action):
        obs,old,terminated,truncated,info=self.env.step(action)
        x=float(info['x']); xd=float(info['x_dot'])
        theta=float(info['theta']); omega=float(info['theta_dot'])
        u=float(info['u'])
        captured=(abs(theta)<=.2 and abs(omega)<=1 and abs(x)<=1.5 and abs(xd)<=.5)
        self.n_capture=self.n_capture+1 if captured else 0
        bonus=0.
        if self.n_capture>=10 and not self.got_capture:
            bonus=25.; self.got_capture=True
        # Ventana continua: centra el carro sólo si el péndulo está suficientemente vertical.
        # Penalizar x lejos de vertical perjudicaría el swing-up y puede provocar rail exits.
        gate=math.exp(-.5*(theta/.20)**2)
        local=gate*(2.0 - 4.0*x*x - .9*xd*xd - 1.6*theta*theta - .16*omega*omega)
        shaped=float(old)+bonus+self.local_weight*local - .02
        info=dict(info);info['reward_p4_local']=self.local_weight*local
        info['capture_bonus_p4']=bonus;info['local_gate_p4']=gate
        return obs,shaped,terminated,truncated,info


class Checkpoint(BaseCallback):
    def __init__(self,folder,every):
        super().__init__();self.folder=Path(folder);self.every=every;self.begin=0
    def _on_training_start(self):self.begin=self.num_timesteps
    def _on_step(self):
        elapsed=self.num_timesteps-self.begin
        if elapsed>0 and elapsed%self.every==0:
            dest=self.folder/f'step_{self.num_timesteps:09d}'
            self.model.save(str(dest))
            self.model.save_replay_buffer(str(dest)+'_replay.pkl')
            print(f'CHECKPOINT {dest}',flush=True)
        return True


def main():
    p=argparse.ArgumentParser(description='P4: entrenamiento DDPG puro para balanceo y carro centrado')
    p.add_argument('--model',required=True,help='Checkpoint P2-40K. No se modifica.')
    p.add_argument('--steps',type=int,default=80000,help='Pasos ADICIONALES de entrenamiento')
    p.add_argument('--seed',type=int,default=43)
    p.add_argument('--checkpoint-every',type=int,default=20000)
    p.add_argument('--upright-probability',type=float,default=.30)
    p.add_argument('--local-weight',type=float,default=1.)
    p.add_argument('--output-root',default='results/optimizacion_ddpg_p4')
    a=p.parse_args()
    if a.steps<6000 or a.checkpoint_every<=0 or not 0<=a.upright_probability<=1 or a.local_weight<0:
        p.error('Revisa steps/checkpoint-every/upright-probability/local-weight')
    model_path=Path(a.model).resolve()
    if not model_path.is_file():p.error(f'No existe modelo: {model_path}')
    output_root=Path(a.output_root).resolve();run=output_root/f'P4_{datetime.now():%Y%m%d_%H%M%S}_seed{a.seed}'
    run.mkdir(parents=True,exist_ok=False);(run/'checkpoints').mkdir()
    def create_env():
        physics=CartPoleContinuousEnv(max_episode_steps=1000,force_max=10,dt=.02,
               noise_std=0,viscous_friction_cart=0,viscous_friction_pole=0,
               random_initial_state=True,start_near_downward=True)
        shaped=P4Env(physics,seed=a.seed,upright_probability=a.upright_probability,local_weight=a.local_weight)
        return Monitor(shaped,filename=str(run/'monitor.csv'))
    env=DummyVecEnv([create_env])
    model=DDPG.load(str(model_path),env=env,device='cpu',print_system_info=False)
    initial=int(model.num_timesteps)
    # Se carga sólo la política: NO hay replay guardado del P2-40K en este paquete.
    # El buffer nuevo debe superar calentamiento antes de cualquier gradiente.
    model.learning_starts=initial+5000
    model.action_noise=NormalActionNoise(mean=np.zeros(1),sigma=.20*np.ones(1))
    # Usar generador no determinista de la carga puede cambiar la semilla; fijar semillas nuevas.
    model.set_random_seed(a.seed)
    meta={'experiment':'P4-v1','initial_policy':str(model_path),'initial_sha256':sha256(model_path),
          'seed':a.seed,'initial_timesteps':initial,'additional_steps':a.steps,
          'upright_probability':a.upright_probability,'local_weight':a.local_weight,
          'reward':'original + one-time capture 25 - .02/step + weight*exp(-.5*(theta/.2)^2)*(2-4*x^2-.9*xdot^2-1.6*theta^2-.16*omega^2)',
          'training_initial_states':'70% downward, 30% randomized near upright by default',
          'replay':'new replay; 5000 new transitions warm-up',
          'dynamics':'unmodified CartPoleContinuousEnv copied from P3-v3',
          'no_classic_controller':True,'sb3':__import__('stable_baselines3').__version__}
    (run/'manifest.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding='utf-8')
    print('P4 START',run,'initial_timesteps',initial,flush=True)
    try:
        model.learn(total_timesteps=a.steps,reset_num_timesteps=False,
                    callback=Checkpoint(run/'checkpoints',a.checkpoint_every),progress_bar=False)
    finally:
        model.save(str(run/'modelo_final_o_interrumpido'))
        model.save_replay_buffer(str(run/'replay_final_o_interrumpido.pkl'))
        env.close()
    print('P4 COMPLETED',run,flush=True)

if __name__=='__main__':main()
