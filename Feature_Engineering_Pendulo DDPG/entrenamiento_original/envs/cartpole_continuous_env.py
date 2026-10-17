"""
Entorno Gymnasium: péndulo invertido tipo carro–poste con acción continua.

Estado:
    s_t = [x_t, x_dot_t, theta_t, theta_dot_t]

Acción:
    a_t = u(t), fuerza continua aplicada al carro.

Convención angular:
    theta = 0  -> péndulo vertical hacia arriba.
    theta = pi -> péndulo hacia abajo.
"""

from __future__ import annotations

import math
from typing import Optional

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class CartPoleContinuousEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        max_episode_steps: int = 1000,
        force_max: float = 10.0,
        dt: float = 0.02,
        x_threshold: float = 2.4,
        noise_std: float = 0.0,
        viscous_friction_cart: float = 0.0,
        viscous_friction_pole: float = 0.0,
        external_impulse: float = 0.0,
        impulse_step: Optional[int] = None,
        random_initial_state: bool = True,
        start_near_downward: bool = True,
    ):
        super().__init__()

        self.gravity = 9.81
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masscart + self.masspole
        self.length = 0.5
        self.polemass_length = self.masspole * self.length

        self.force_max = float(force_max)
        self.dt = float(dt)
        self.x_threshold = float(x_threshold)
        self.max_episode_steps = int(max_episode_steps)

        self.noise_std = float(noise_std)
        self.viscous_friction_cart = float(viscous_friction_cart)
        self.viscous_friction_pole = float(viscous_friction_pole)
        self.external_impulse = float(external_impulse)
        self.impulse_step = impulse_step

        self.random_initial_state = random_initial_state
        self.start_near_downward = start_near_downward

        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.screen_width = 800
        self.screen_height = 500

        self.action_space = spaces.Box(
            low=np.array([-self.force_max], dtype=np.float32),
            high=np.array([self.force_max], dtype=np.float32),
            dtype=np.float32,
        )

        high = np.array(
            [
                self.x_threshold * 2.0,
                np.finfo(np.float32).max,
                math.pi,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.state: Optional[np.ndarray] = None
        self.steps = 0
        self.last_u = 0.0
        self.cumulative_control_effort = 0.0
        self.cumulative_reward = 0.0

    @staticmethod
    def angle_normalize(theta: float) -> float:
        return ((theta + math.pi) % (2.0 * math.pi)) - math.pi

    def _get_obs(self) -> np.ndarray:
        assert self.state is not None
        obs = self.state.copy()

        if self.noise_std > 0:
            noise = self.np_random.normal(0.0, self.noise_std, size=obs.shape)
            obs = obs + noise.astype(np.float32)

        obs[2] = self.angle_normalize(float(obs[2]))
        return obs.astype(np.float32)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.steps = 0
        self.last_u = 0.0
        self.cumulative_control_effort = 0.0
        self.cumulative_reward = 0.0

        if self.random_initial_state:
            if self.start_near_downward:
                x = self.np_random.uniform(low=-0.05, high=0.05)
                x_dot = self.np_random.uniform(low=-0.05, high=0.05)
                theta = math.pi + self.np_random.uniform(low=-0.10, high=0.10)
                theta_dot = self.np_random.uniform(low=-0.05, high=0.05)
            else:
                x = self.np_random.uniform(low=-0.5, high=0.5)
                x_dot = self.np_random.uniform(low=-0.5, high=0.5)
                theta = self.np_random.uniform(low=-math.pi, high=math.pi)
                theta_dot = self.np_random.uniform(low=-1.0, high=1.0)
        else:
            x, x_dot, theta, theta_dot = 0.0, 0.0, math.pi, 0.0

        theta = self.angle_normalize(theta)
        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)

        info = {
            "cumulative_control_effort": self.cumulative_control_effort,
            "cumulative_reward": self.cumulative_reward,
        }
        return self._get_obs(), info

    def step(self, action):
        assert self.state is not None

        x, x_dot, theta, theta_dot = map(float, self.state)
        force = float(np.clip(action[0], -self.force_max, self.force_max))

        if self.impulse_step is not None and self.steps == self.impulse_step:
            force += self.external_impulse

        force -= self.viscous_friction_cart * x_dot

        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        temp = (
            force + self.polemass_length * theta_dot**2 * sintheta
        ) / self.total_mass

        theta_acc = (
            self.gravity * sintheta
            - costheta * temp
            - self.viscous_friction_pole * theta_dot / max(self.polemass_length, 1e-8)
        ) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )

        x_acc = temp - self.polemass_length * theta_acc * costheta / self.total_mass

        x_dot = x_dot + self.dt * x_acc
        x = x + self.dt * x_dot
        theta_dot = theta_dot + self.dt * theta_acc
        theta = theta + self.dt * theta_dot
        theta = self.angle_normalize(theta)

        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)
        self.steps += 1

        reward = self._reward(x, x_dot, theta, theta_dot, force)

        terminated = bool(abs(x) > self.x_threshold)
        truncated = bool(self.steps >= self.max_episode_steps)

        self.last_u = force
        self.cumulative_control_effort += force**2 * self.dt
        self.cumulative_reward += reward

        info = {
            "x": x,
            "x_dot": x_dot,
            "theta": theta,
            "theta_dot": theta_dot,
            "u": force,
            "cumulative_control_effort": self.cumulative_control_effort,
            "cumulative_reward": self.cumulative_reward,
            "is_upright": abs(theta) < 0.15,
        }

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), float(reward), terminated, truncated, info

    def _reward(self, x: float, x_dot: float, theta: float, theta_dot: float, u: float) -> float:
        """
        Recompensa mejorada para swing-up y estabilización.
        """

        reward = 1.0 + math.cos(theta)

        reward += 0.05

        reward -= 0.20 * x**2
        reward -= 0.01 * x_dot**2
        reward -= 0.02 * theta_dot**2
        reward -= 0.001 * u**2

        if abs(theta) < 0.15:
            reward += 3.0

        if abs(x) > 0.80 * self.x_threshold:
            reward -= 5.0 * (abs(x) - 0.80 * self.x_threshold)

        if abs(x) > self.x_threshold:
            reward -= 100.0

        return reward

    def render(self):
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as exc:
            raise ImportError("Para renderizar instala pygame: pip install pygame") from exc

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            else:
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
            self.clock = pygame.time.Clock()

        assert self.state is not None
        x, _, theta, _ = self.state

        world_width = self.x_threshold * 2
        scale = self.screen_width / world_width
        cart_y = int(self.screen_height * 0.65)
        cart_w = 60
        cart_h = 30
        pole_len = scale * (2 * self.length)

        cart_x = int(self.screen_width / 2 + x * scale)

        self.screen.fill((255, 255, 255))
        import pygame

        pygame.draw.line(
            self.screen, (0, 0, 0),
            (0, cart_y + cart_h // 2),
            (self.screen_width, cart_y + cart_h // 2),
            2
        )

        cart_rect = pygame.Rect(cart_x - cart_w // 2, cart_y - cart_h // 2, cart_w, cart_h)
        pygame.draw.rect(self.screen, (30, 144, 255), cart_rect)
        pygame.draw.rect(self.screen, (0, 0, 0), cart_rect, 2)

        pivot = (cart_x, cart_y - cart_h // 2)
        pole_x = pivot[0] + pole_len * math.sin(theta)
        pole_y = pivot[1] - pole_len * math.cos(theta)

        pygame.draw.line(self.screen, (220, 80, 20), pivot, (pole_x, pole_y), 6)
        gfxdraw.filled_circle(self.screen, int(pole_x), int(pole_y), 10, (220, 80, 20))
        gfxdraw.aacircle(self.screen, int(pole_x), int(pole_y), 10, (0, 0, 0))

        font = pygame.font.SysFont("Arial", 18)
        txt = font.render(f"theta={theta:.2f} rad | u={self.last_u:.2f} N", True, (0, 0, 0))
        self.screen.blit(txt, (20, 20))

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.screen = None
            self.clock = None
