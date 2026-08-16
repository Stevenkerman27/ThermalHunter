import os

import gymnasium as gym
from gymnasium import spaces
import numpy as np

import config
from glider_discrete_simp import RBWindField, GliderPhysics, compute_wind_amplification


class DynamicGliderEnv(gym.Env):
    """Non-steady point-mass glider environment with continuous controls."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        h5_paths=None,
        polar_file_base=config.POLAR_BASE,
        domain_size=config.DOMAIN_SIZE,
        memory_mode=False,
        wind_manager=None,
    ):
        super().__init__()
        self.domain_size = np.asarray(domain_size, dtype=np.float64)
        if wind_manager is None:
            if h5_paths is None:
                raise ValueError("h5_paths is required when wind_manager is not supplied")
            self.wind_manager = RBWindField(h5_paths, domain_size=domain_size, memory_mode=memory_mode)
            self._owns_wind_manager = True
        else:
            if h5_paths is not None:
                raise ValueError("provide either h5_paths or wind_manager, not both")
            self.wind_manager = wind_manager
            self._owns_wind_manager = False
        self.physics = GliderPhysics(polar_file_base)
        self.wind_ampf = compute_wind_amplification(self.wind_manager, self.physics)

        self.dt_rl = config.DT_RL
        self.dt_integration = config.DYNAMIC_DT_INTEGRATION
        self.n_integration_steps = int(round(self.dt_rl / self.dt_integration))
        if not np.isclose(self.n_integration_steps * self.dt_integration, self.dt_rl):
            raise ValueError("DT_RL must be an integer multiple of DYNAMIC_DT_INTEGRATION")

        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.full(2, -np.inf, dtype=np.float32),
            high=np.full(2, np.inf, dtype=np.float32),
            dtype=np.float32,
        )

        self.alpha_min = np.deg2rad(config.DYNAMIC_AOA_MIN_DEG)
        self.alpha_max = np.deg2rad(config.DYNAMIC_AOA_MAX_DEG)
        self.alpha_rate_limit = np.deg2rad(config.DYNAMIC_AOA_RATE_LIMIT_DEG_S)
        self.bank_min = np.deg2rad(config.BANK_MIN_DEG)
        self.bank_max = np.deg2rad(config.BANK_MAX_DEG)
        self.bank_rate_limit = np.deg2rad(config.DYNAMIC_BANK_RATE_LIMIT_DEG_S)
        self.gravity = np.array([0.0, 0.0, -self.physics.g], dtype=np.float64)

    def _wind(self, position):
        return self.wind_manager.get_wind_at_frame(self.wind_frame, *position) * self.wind_ampf

    @staticmethod
    def _normalized(vector):
        magnitude = np.linalg.norm(vector)
        if magnitude == 0.0:
            raise ValueError("cannot normalize a zero vector")
        return vector / magnitude

    def _update_controls(self, action, dt):
        alpha_command = self.alpha_max - action[0] * (self.alpha_max - self.alpha_min)
        alpha_rate = np.clip(
            (alpha_command - self.alpha) / config.DYNAMIC_AOA_TIME_CONSTANT,
            -self.alpha_rate_limit,
            self.alpha_rate_limit,
        )
        self.alpha = np.clip(self.alpha + alpha_rate * dt, self.alpha_min, self.alpha_max)

        bank_command = self.bank_min + action[1] * (self.bank_max - self.bank_min)
        bank_rate = np.clip(
            (bank_command - self.bank) / config.DYNAMIC_BANK_TIME_CONSTANT,
            -self.bank_rate_limit,
            self.bank_rate_limit,
        )
        self.bank = np.clip(self.bank + bank_rate * dt, self.bank_min, self.bank_max)

    def _aerodynamic_acceleration(self, wind):
        air_velocity = self.ground_velocity - wind
        tas = np.linalg.norm(air_velocity)
        if tas < config.DYNAMIC_MIN_TAS:
            return np.zeros(3, dtype=np.float64), tas, None, None

        forward = air_velocity / tas
        right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
        if np.linalg.norm(right) < 1e-8:
            right = np.cross(forward, np.array([0.0, 1.0, 0.0]))
        right = self._normalized(right)
        lift_up = self._normalized(np.cross(right, forward))
        lift_direction = np.cos(self.bank) * lift_up + np.sin(self.bank) * right

        cl = float(self.physics.aero_interp["Cl"](np.degrees(self.alpha)))
        cd = float(self.physics.aero_interp["Cd"](np.degrees(self.alpha)))
        dynamic_pressure = 0.5 * self.physics.rho * tas * tas
        lift = dynamic_pressure * self.physics.A * cl
        drag = dynamic_pressure * self.physics.A * cd
        acceleration = (lift * lift_direction - drag * forward) / self.physics.m + self.gravity
        return acceleration, tas, right, lift_direction

    def _wingtip_normal_wind_difference(self, right, lift_direction):
        half_span = config.WINGSPAN / 2.0
        right_tip = self.position + half_span * right
        left_tip = self.position - half_span * right
        right_tip[:2] %= self.domain_size[:2]
        left_tip[:2] %= self.domain_size[:2]
        right_tip[2] = np.clip(right_tip[2], 0.0, self.domain_size[2] - 0.01)
        left_tip[2] = np.clip(left_tip[2], 0.0, self.domain_size[2] - 0.01)
        return float(np.dot(self._wind(right_tip) - self._wind(left_tip), lift_direction))

    def _energy_height(self, tas):
        return self.position[2] + 0.5 * tas * tas / self.physics.g

    def _get_obs(self):
        return np.array([self.vario, self.roll_cue], dtype=np.float32)

    def _info(self, energy_change=0.0):
        air_velocity = self.ground_velocity - self._wind(self.position)
        tas = float(np.linalg.norm(air_velocity))
        energy_height = self._energy_height(tas)
        return {
            "height": float(self.position[2]),
            "initial_height": float(self.initial_height),
            "tas": tas,
            "alpha_deg": float(np.degrees(self.alpha)),
            "bank_deg": float(np.degrees(self.bank)),
            "energy_height": float(energy_height),
            "initial_energy_height": float(self.initial_energy_height),
            "energy_change": float(energy_change),
            "total_energy_vario": float(self.vario),
            "roll_cue": float(self.roll_cue),
            "wind_frame": float(self.wind_frame),
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        options = {} if options is None else options
        reset_frame = options["resettime"] if "resettime" in options else config.sample_start_frame(self.np_random)
        if not config.RESET_START_MIN <= reset_frame < config.RESET_START_MAX:
            raise ValueError("resettime must use the configured stable-frame sampling range")
        if reset_frame > self.wind_manager.max_t_idx:
            raise ValueError("resettime exceeds available wind-field frames")
        self.wind_frame = float(reset_frame)

        if "initial_position" in options:
            self.position = np.asarray(options["initial_position"], dtype=np.float64).copy()
        else:
            x, y = self.np_random.uniform(0.2, 0.8, size=2) * self.domain_size[:2]
            z = self.np_random.uniform(0.2, 0.6) * self.domain_size[2]
            self.position = np.array([x, y, z], dtype=np.float64)
        self.position[:2] %= self.domain_size[:2]

        heading = float(options.get("initial_heading", self.np_random.uniform(0.0, 2.0 * np.pi)))
        self.alpha = 0.5 * (self.alpha_min + self.alpha_max)
        self.bank = 0.0
        trim_tas, trim_gamma, _ = self.physics.get_steady_state(self.alpha, self.bank)
        air_velocity = np.array(
            [
                trim_tas * np.cos(trim_gamma) * np.cos(heading),
                trim_tas * np.cos(trim_gamma) * np.sin(heading),
                -trim_tas * np.sin(trim_gamma),
            ],
            dtype=np.float64,
        )
        self.ground_velocity = air_velocity + self._wind(self.position)
        self.vario = 0.0
        self.roll_cue = 0.0
        self.initial_height = float(self.position[2])
        initial_tas = float(np.linalg.norm(air_velocity))
        self.initial_energy_height = self._energy_height(initial_tas)
        self._last_energy_height = self.initial_energy_height
        return self._get_obs(), self._info()

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (2,):
            raise ValueError("dynamic action must have shape (2,)")
        action = np.clip(action, 0.0, 1.0)
        starting_energy_height = self._energy_height(np.linalg.norm(self.ground_velocity - self._wind(self.position)))
        roll_cue_sum = 0.0
        terminated = False
        truncated = False

        for _ in range(self.n_integration_steps):
            self._update_controls(action, self.dt_integration)
            wind = self._wind(self.position)
            acceleration, tas, right, lift_direction = self._aerodynamic_acceleration(wind)
            if tas < config.DYNAMIC_MIN_TAS:
                terminated = True
                break

            self.ground_velocity += acceleration * self.dt_integration
            self.position += self.ground_velocity * self.dt_integration
            self.position[:2] %= self.domain_size[:2]
            self.wind_frame += self.dt_integration / config.DYNAMIC_WIND_SECONDS_PER_FRAME

            new_tas = float(np.linalg.norm(self.ground_velocity - self._wind(self.position)))
            energy_height = self._energy_height(new_tas)
            raw_vario = (energy_height - starting_energy_height) / self.dt_integration
            filter_gain = 1.0 - np.exp(-self.dt_integration / config.DYNAMIC_VARIO_TIME_CONSTANT)
            self.vario += filter_gain * (raw_vario - self.vario)
            roll_cue_sum += self._wingtip_normal_wind_difference(right, lift_direction)
            starting_energy_height = energy_height

            if self.position[2] <= self.domain_size[2] * config.DYNAMIC_ALTITUDE_MIN_FRACTION:
                terminated = True
                break
            if self.position[2] >= self.domain_size[2] * config.DYNAMIC_ALTITUDE_MAX_FRACTION:
                terminated = True
                break
            if self.wind_frame >= self.wind_manager.max_t_idx:
                truncated = True
                break

        self.roll_cue = roll_cue_sum / self.n_integration_steps
        ending_tas = float(np.linalg.norm(self.ground_velocity - self._wind(self.position)))
        ending_energy_height = self._energy_height(ending_tas)
        reward = ending_energy_height - self._last_energy_height
        self._last_energy_height = ending_energy_height
        return self._get_obs(), float(reward), terminated, truncated, self._info(reward)

    def close(self):
        if self._owns_wind_manager:
            self.wind_manager.close()


class DynamicDiscreteActionWrapper(gym.ActionWrapper):
    """Map a Cartesian grid of speed and roll commands onto dynamic controls."""

    def __init__(self, env, action_levels=config.DYNAMIC_DQN_ACTION_LEVELS):
        super().__init__(env)
        if action_levels < 2:
            raise ValueError("dynamic DQN requires at least two action levels")
        self.action_levels = action_levels
        self.command_values = np.linspace(0.0, 1.0, action_levels, dtype=np.float32)
        self.action_space = spaces.Discrete(action_levels * action_levels)

    def action(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"dynamic discrete action must be in [0, {self.action_space.n})")
        action_index = int(action)
        speed_index, roll_index = divmod(action_index, self.action_levels)
        return np.array(
            [self.command_values[speed_index], self.command_values[roll_index]],
            dtype=np.float32,
        )

    def command_to_action(self, command):
        command = np.asarray(command, dtype=np.float32)
        if command.shape != (2,) or np.any(command < 0.0) or np.any(command > 1.0):
            raise ValueError("dynamic command must have shape (2,) and values in [0, 1]")
        indices = np.rint(command * (self.action_levels - 1)).astype(int)
        return int(indices[0] * self.action_levels + indices[1])
