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
            low=np.full(4, -np.inf, dtype=np.float32),
            high=np.full(4, np.inf, dtype=np.float32),
            dtype=np.float32,
        )

        self.alpha_min = np.deg2rad(config.DYNAMIC_AOA_MIN_DEG)
        self.alpha_max = np.deg2rad(config.DYNAMIC_AOA_MAX_DEG)
        self.alpha_rate_limit = np.deg2rad(config.DYNAMIC_AOA_RATE_LIMIT_DEG_S)
        self.bank_min = np.deg2rad(config.BANK_MIN_DEG)
        self.bank_max = np.deg2rad(config.BANK_MAX_DEG)
        self.bank_rate_limit = np.deg2rad(config.DYNAMIC_BANK_RATE_LIMIT_DEG_S)
        self.gravity = np.array([0.0, 0.0, -self.physics.g], dtype=np.float64)

    def _wind(self, position, wind_frame=None):
        wind_position = np.asarray(position, dtype=np.float64).copy()
        wind_position[:2] %= self.domain_size[:2]
        frame = self.wind_frame if wind_frame is None else wind_frame
        return self.wind_manager.get_wind_at_frame(frame, *wind_position) * self.wind_ampf

    @staticmethod
    def _normalized(vector):
        magnitude = np.linalg.norm(vector)
        if magnitude == 0.0:
            raise ValueError("cannot normalize a zero vector")
        return vector / magnitude

    def _control_rates(self, action, alpha, bank):
        alpha_command = self.alpha_max - action[0] * (self.alpha_max - self.alpha_min)
        alpha_rate = np.clip(
            (alpha_command - alpha) / config.DYNAMIC_AOA_TIME_CONSTANT,
            -self.alpha_rate_limit,
            self.alpha_rate_limit,
        )
        bank_command = self.bank_min + action[1] * (self.bank_max - self.bank_min)
        bank_rate = np.clip(
            (bank_command - bank) / config.DYNAMIC_BANK_TIME_CONSTANT,
            -self.bank_rate_limit,
            self.bank_rate_limit,
        )
        return alpha_rate, bank_rate

    def _aerodynamic_acceleration(self, wind, ground_velocity, alpha, bank):
        air_velocity = ground_velocity - wind
        tas = np.linalg.norm(air_velocity)
        if tas < config.DYNAMIC_MIN_TAS:
            # Low airspeed invalidates the aerodynamic-force model, not gravity.
            return self.gravity.copy(), tas, None, None

        forward = air_velocity / tas
        right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
        if np.linalg.norm(right) < 1e-8:
            right = np.cross(forward, np.array([0.0, 1.0, 0.0]))
        right = self._normalized(right)
        lift_up = self._normalized(np.cross(right, forward))
        lift_direction = np.cos(bank) * lift_up + np.sin(bank) * right

        cl = float(self.physics.aero_interp["Cl"](np.degrees(alpha)))
        cd = float(self.physics.aero_interp["Cd"](np.degrees(alpha)))
        dynamic_pressure = 0.5 * self.physics.rho * tas * tas
        lift = dynamic_pressure * self.physics.A * cl
        drag = dynamic_pressure * self.physics.A * cd
        acceleration = (lift * lift_direction - drag * forward) / self.physics.m + self.gravity
        return acceleration, tas, right, lift_direction

    def _wingtip_normal_wind_difference(self, position, wind_frame, right, lift_direction):
        half_span = config.WINGSPAN / 2.0
        right_tip = position + half_span * right
        left_tip = position - half_span * right
        right_tip[2] = np.clip(right_tip[2], 0.0, self.domain_size[2] - 0.01)
        left_tip[2] = np.clip(left_tip[2], 0.0, self.domain_size[2] - 0.01)
        return float(
            np.dot(
                self._wind(right_tip, wind_frame) - self._wind(left_tip, wind_frame),
                lift_direction,
            )
        )

    def _state_vector(self):
        return np.concatenate(
            [
                self.position,
                self.ground_velocity,
                np.array([self.alpha, self.bank, self.wind_frame], dtype=np.float64),
            ]
        )

    def _set_state_vector(self, state):
        self.position = state[:3].copy()
        self.position[:2] %= self.domain_size[:2]
        self.ground_velocity = state[3:6].copy()
        self.alpha = float(np.clip(state[6], self.alpha_min, self.alpha_max))
        self.bank = float(np.clip(state[7], self.bank_min, self.bank_max))
        self.wind_frame = float(state[8])

    def _state_derivative(self, state, action):
        position = state[:3]
        ground_velocity = state[3:6]
        alpha = state[6]
        bank = state[7]
        wind_frame = state[8]
        wind = self._wind(position, wind_frame)
        acceleration, _, _, _ = self._aerodynamic_acceleration(wind, ground_velocity, alpha, bank)
        alpha_rate, bank_rate = self._control_rates(action, alpha, bank)
        return np.concatenate(
            [
                ground_velocity,
                acceleration,
                np.array(
                    [alpha_rate, bank_rate, 1.0 / config.DYNAMIC_WIND_SECONDS_PER_FRAME],
                    dtype=np.float64,
                ),
            ]
        )

    def _rk4_step(self, action):
        state = self._state_vector()
        dt = self.dt_integration
        k1 = self._state_derivative(state, action)
        state_k2 = state + 0.5 * dt * k1
        if not self._is_numerically_valid_state(state_k2):
            return None
        k2 = self._state_derivative(state_k2, action)
        state_k3 = state + 0.5 * dt * k2
        if not self._is_numerically_valid_state(state_k3):
            return None
        k3 = self._state_derivative(state_k3, action)
        state_k4 = state + dt * k3
        if not self._is_numerically_valid_state(state_k4):
            return None
        k4 = self._state_derivative(state_k4, action)
        candidate_state = state + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        return candidate_state if self._is_numerically_valid_state(candidate_state) else None

    def _is_numerically_valid_state(self, state):
        if not np.all(np.isfinite(state)):
            return False
        wind = self._wind(state[:3], state[8])
        tas = float(np.linalg.norm(state[3:6] - wind))
        return np.isfinite(tas) and tas <= config.DYNAMIC_NUMERICAL_MAX_TAS

    def _energy_height(self, tas):
        return self.position[2] + 0.5 * tas * tas / self.physics.g

    def _get_obs(self):
        air_velocity = self.ground_velocity - self._wind(self.position)
        energy_height = self._energy_height(np.linalg.norm(air_velocity))
        return np.array(
            [energy_height, self.vario, self.roll_cue, self.bank],
            dtype=np.float32,
        )

    def _info(self, energy_change=0.0, termination_reason=None):
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
            "termination_reason": termination_reason,
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
        termination_reason = None

        for _ in range(self.n_integration_steps):
            candidate_state = self._rk4_step(action)
            if candidate_state is None:
                terminated = True
                termination_reason = "numerical_divergence"
                break
            self._set_state_vector(candidate_state)

            new_tas = float(np.linalg.norm(self.ground_velocity - self._wind(self.position)))
            energy_height = self._energy_height(new_tas)
            raw_vario = (energy_height - starting_energy_height) / self.dt_integration
            filter_gain = 1.0 - np.exp(-self.dt_integration / config.DYNAMIC_VARIO_TIME_CONSTANT)
            self.vario += filter_gain * (raw_vario - self.vario)
            wind = self._wind(self.position)
            _, _, right, lift_direction = self._aerodynamic_acceleration(
                wind, self.ground_velocity, self.alpha, self.bank
            )
            if right is not None:
                roll_cue_sum += self._wingtip_normal_wind_difference(
                    self.position, self.wind_frame, right, lift_direction
                )
            starting_energy_height = energy_height

            if self.position[2] <= self.domain_size[2] * config.DYNAMIC_ALTITUDE_MIN_FRACTION:
                terminated = True
                termination_reason = "altitude_low"
                break
            if self.position[2] >= self.domain_size[2] * config.DYNAMIC_ALTITUDE_MAX_FRACTION:
                terminated = True
                termination_reason = "altitude_high"
                break
            if self.wind_frame >= self.wind_manager.max_t_idx:
                truncated = True
                termination_reason = "wind_end"
                break

        self.roll_cue = roll_cue_sum / self.n_integration_steps
        ending_tas = float(np.linalg.norm(self.ground_velocity - self._wind(self.position)))
        ending_energy_height = self._energy_height(ending_tas)
        reward = 0.0 if termination_reason == "numerical_divergence" else ending_energy_height - self._last_energy_height
        self._last_energy_height = ending_energy_height
        return self._get_obs(), float(reward), terminated, truncated, self._info(reward, termination_reason)

    def close(self):
        if self._owns_wind_manager:
            self.wind_manager.close()


class DynamicGliderBatchEnv:
    """Synchronously integrate many dynamic gliders against one shared wind field."""

    def __init__(
        self,
        num_envs,
        h5_paths=None,
        polar_file_base=config.POLAR_BASE,
        domain_size=config.DOMAIN_SIZE,
        memory_mode=True,
        wind_manager=None,
        autoreset=True,
    ):
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        self.num_envs = int(num_envs)
        self.domain_size = np.asarray(domain_size, dtype=np.float64)
        if wind_manager is None:
            if h5_paths is None:
                raise ValueError("h5_paths is required when wind_manager is not supplied")
            self.wind_manager = RBWindField(h5_paths, domain_size=domain_size, memory_mode=memory_mode)
            self._owns_wind_manager = True
        else:
            if h5_paths is not None:
                raise ValueError("provide either h5_paths or wind_manager, not both")
            if not wind_manager.memory_mode:
                raise ValueError("DynamicGliderBatchEnv requires a memory-resident wind field")
            self.wind_manager = wind_manager
            self._owns_wind_manager = False
        if not self.wind_manager.memory_mode:
            raise ValueError("DynamicGliderBatchEnv requires memory_mode=True")

        self.physics = GliderPhysics(polar_file_base)
        self.wind_ampf = compute_wind_amplification(self.wind_manager, self.physics)
        self.dt_rl = config.DT_RL
        self.dt_integration = config.DYNAMIC_DT_INTEGRATION
        self.n_integration_steps = int(round(self.dt_rl / self.dt_integration))
        if not np.isclose(self.n_integration_steps * self.dt_integration, self.dt_rl):
            raise ValueError("DT_RL must be an integer multiple of DYNAMIC_DT_INTEGRATION")

        self.single_action_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
        self.single_observation_space = spaces.Box(
            low=np.full(4, -np.inf, dtype=np.float32),
            high=np.full(4, np.inf, dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = self.single_action_space
        self.observation_space = self.single_observation_space
        self.alpha_min = np.deg2rad(config.DYNAMIC_AOA_MIN_DEG)
        self.alpha_max = np.deg2rad(config.DYNAMIC_AOA_MAX_DEG)
        self.alpha_rate_limit = np.deg2rad(config.DYNAMIC_AOA_RATE_LIMIT_DEG_S)
        self.bank_min = np.deg2rad(config.BANK_MIN_DEG)
        self.bank_max = np.deg2rad(config.BANK_MAX_DEG)
        self.bank_rate_limit = np.deg2rad(config.DYNAMIC_BANK_RATE_LIMIT_DEG_S)
        self.gravity = np.array([0.0, 0.0, -self.physics.g], dtype=np.float64)
        self.autoreset = autoreset
        self.np_random = np.random.default_rng()
        self._done = np.zeros(self.num_envs, dtype=bool)

    def _wind(self, positions, wind_frames):
        wind_positions = np.asarray(positions, dtype=np.float64).copy()
        wind_positions[:, :2] %= self.domain_size[:2]
        return self.wind_manager.get_winds_at_frames(wind_frames, wind_positions) * self.wind_ampf

    def _control_rates(self, actions, alpha, bank):
        alpha_command = self.alpha_max - actions[:, 0] * (self.alpha_max - self.alpha_min)
        alpha_rate = np.clip(
            (alpha_command - alpha) / config.DYNAMIC_AOA_TIME_CONSTANT,
            -self.alpha_rate_limit,
            self.alpha_rate_limit,
        )
        bank_command = self.bank_min + actions[:, 1] * (self.bank_max - self.bank_min)
        bank_rate = np.clip(
            (bank_command - bank) / config.DYNAMIC_BANK_TIME_CONSTANT,
            -self.bank_rate_limit,
            self.bank_rate_limit,
        )
        return alpha_rate, bank_rate

    def _aerodynamic_acceleration(self, wind, ground_velocity, alpha, bank):
        air_velocity = ground_velocity - wind
        tas = np.linalg.norm(air_velocity, axis=1)
        active = tas >= config.DYNAMIC_MIN_TAS
        acceleration = np.broadcast_to(self.gravity, ground_velocity.shape).copy()
        right = np.zeros_like(ground_velocity)
        lift_direction = np.zeros_like(ground_velocity)
        if not np.any(active):
            return acceleration, tas, right, lift_direction, active

        forward = air_velocity[active] / tas[active, None]
        active_right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
        fallback = np.linalg.norm(active_right, axis=1) < 1e-8
        active_right[fallback] = np.cross(forward[fallback], np.array([0.0, 1.0, 0.0]))
        right_norm = np.linalg.norm(active_right, axis=1)
        if np.any(right_norm == 0.0):
            raise ValueError("cannot normalize a zero vector")
        active_right /= right_norm[:, None]
        lift_up = np.cross(active_right, forward)
        lift_norm = np.linalg.norm(lift_up, axis=1)
        if np.any(lift_norm == 0.0):
            raise ValueError("cannot normalize a zero vector")
        lift_up /= lift_norm[:, None]
        active_lift_direction = np.cos(bank[active, None]) * lift_up + np.sin(bank[active, None]) * active_right
        cl = self.physics.aero_interp["Cl"](np.degrees(alpha[active]))
        cd = self.physics.aero_interp["Cd"](np.degrees(alpha[active]))
        dynamic_pressure = 0.5 * self.physics.rho * tas[active] * tas[active]
        lift = dynamic_pressure * self.physics.A * cl
        drag = dynamic_pressure * self.physics.A * cd
        acceleration[active] = (
            lift[:, None] * active_lift_direction - drag[:, None] * forward
        ) / self.physics.m + self.gravity
        right[active] = active_right
        lift_direction[active] = active_lift_direction
        return acceleration, tas, right, lift_direction, active

    def _state_derivative(self, states, actions, mask):
        derivative = np.zeros_like(states)
        if not np.any(mask):
            return derivative
        selected = np.flatnonzero(mask)
        state = states[selected]
        wind = self._wind(state[:, :3], state[:, 8])
        acceleration, _, _, _, _ = self._aerodynamic_acceleration(
            wind, state[:, 3:6], state[:, 6], state[:, 7]
        )
        alpha_rate, bank_rate = self._control_rates(actions[selected], state[:, 6], state[:, 7])
        derivative[selected, :3] = state[:, 3:6]
        derivative[selected, 3:6] = acceleration
        derivative[selected, 6] = alpha_rate
        derivative[selected, 7] = bank_rate
        derivative[selected, 8] = 1.0 / config.DYNAMIC_WIND_SECONDS_PER_FRAME
        return derivative

    def _valid_states(self, states, mask):
        valid = np.zeros(self.num_envs, dtype=bool)
        tas = np.full(self.num_envs, np.nan, dtype=np.float64)
        finite = mask & np.isfinite(states).all(axis=1)
        if np.any(finite):
            selected = np.flatnonzero(finite)
            wind = self._wind(states[selected, :3], states[selected, 8])
            tas[selected] = np.linalg.norm(states[selected, 3:6] - wind, axis=1)
            valid[selected] = np.isfinite(tas[selected]) & (tas[selected] <= config.DYNAMIC_NUMERICAL_MAX_TAS)
        return valid, tas

    def _rk4_candidates(self, actions, alive):
        state = self._state_matrix()
        dt = self.dt_integration
        k1 = self._state_derivative(state, actions, alive)
        state_k2 = state + 0.5 * dt * k1
        valid_k2, _ = self._valid_states(state_k2, alive)
        k2 = self._state_derivative(state_k2, actions, valid_k2)
        state_k3 = state + 0.5 * dt * k2
        valid_k3, _ = self._valid_states(state_k3, valid_k2)
        k3 = self._state_derivative(state_k3, actions, valid_k3)
        state_k4 = state + dt * k3
        valid_k4, _ = self._valid_states(state_k4, valid_k3)
        k4 = self._state_derivative(state_k4, actions, valid_k4)
        candidate = state + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        valid_candidate, _ = self._valid_states(candidate, valid_k4)
        return candidate, valid_candidate

    def _state_matrix(self):
        return np.column_stack(
            [self.position, self.ground_velocity, self.alpha, self.bank, self.wind_frame]
        )

    def _set_states(self, states, mask):
        selected = np.flatnonzero(mask)
        if len(selected) == 0:
            return
        self.position[selected] = states[selected, :3]
        self.position[selected, :2] %= self.domain_size[:2]
        self.ground_velocity[selected] = states[selected, 3:6]
        self.alpha[selected] = np.clip(states[selected, 6], self.alpha_min, self.alpha_max)
        self.bank[selected] = np.clip(states[selected, 7], self.bank_min, self.bank_max)
        self.wind_frame[selected] = states[selected, 8]

    def _observations(self):
        wind = self._wind(self.position, self.wind_frame)
        tas = np.linalg.norm(self.ground_velocity - wind, axis=1)
        energy_height = self.position[:, 2] + 0.5 * tas * tas / self.physics.g
        return np.column_stack([energy_height, self.vario, self.roll_cue, self.bank]).astype(np.float32)

    def _infos(self, energy_change, termination_reasons):
        wind = self._wind(self.position, self.wind_frame)
        tas = np.linalg.norm(self.ground_velocity - wind, axis=1)
        energy_height = self.position[:, 2] + 0.5 * tas * tas / self.physics.g
        return {
            "height": self.position[:, 2].copy(),
            "initial_height": self.initial_height.copy(),
            "tas": tas,
            "alpha_deg": np.degrees(self.alpha),
            "bank_deg": np.degrees(self.bank),
            "energy_height": energy_height,
            "initial_energy_height": self.initial_energy_height.copy(),
            "energy_change": np.asarray(energy_change, dtype=np.float64).copy(),
            "total_energy_vario": self.vario.copy(),
            "roll_cue": self.roll_cue.copy(),
            "wind_frame": self.wind_frame.copy(),
            "termination_reason": termination_reasons.copy(),
        }

    def _normalized_options(self, options):
        if options is None:
            return [None] * self.num_envs
        if not isinstance(options, (list, tuple)) or len(options) != self.num_envs:
            raise ValueError("batched reset options must be a sequence with one entry per environment")
        return list(options)

    def _reset_slots(self, indices, options):
        for index, option in zip(indices, options):
            option = {} if option is None else option
            reset_frame = option["resettime"] if "resettime" in option else config.sample_start_frame(self.np_random)
            if not config.RESET_START_MIN <= reset_frame < config.RESET_START_MAX:
                raise ValueError("resettime must use the configured stable-frame sampling range")
            if reset_frame > self.wind_manager.max_t_idx:
                raise ValueError("resettime exceeds available wind-field frames")
            self.wind_frame[index] = float(reset_frame)
            if "initial_position" in option:
                position = np.asarray(option["initial_position"], dtype=np.float64)
                if position.shape != (3,):
                    raise ValueError("initial_position must have shape (3,)")
                self.position[index] = position
            else:
                x, y = self.np_random.uniform(0.2, 0.8, size=2) * self.domain_size[:2]
                self.position[index] = [x, y, self.np_random.uniform(0.2, 0.6) * self.domain_size[2]]
            self.position[index, :2] %= self.domain_size[:2]
            heading = float(option.get("initial_heading", self.np_random.uniform(0.0, 2.0 * np.pi)))
            self.alpha[index] = 0.5 * (self.alpha_min + self.alpha_max)
            self.bank[index] = 0.0
            trim_tas, trim_gamma, _ = self.physics.get_steady_state(self.alpha[index], self.bank[index])
            air_velocity = np.array(
                [
                    trim_tas * np.cos(trim_gamma) * np.cos(heading),
                    trim_tas * np.cos(trim_gamma) * np.sin(heading),
                    -trim_tas * np.sin(trim_gamma),
                ],
                dtype=np.float64,
            )
            wind = self._wind(self.position[index:index + 1], self.wind_frame[index:index + 1])[0]
            self.ground_velocity[index] = air_velocity + wind
            self.vario[index] = 0.0
            self.roll_cue[index] = 0.0
            self.initial_height[index] = self.position[index, 2]
            self.initial_energy_height[index] = self.position[index, 2] + 0.5 * trim_tas * trim_tas / self.physics.g
            self._last_energy_height[index] = self.initial_energy_height[index]
            self._done[index] = False

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        self.position = np.empty((self.num_envs, 3), dtype=np.float64)
        self.ground_velocity = np.empty((self.num_envs, 3), dtype=np.float64)
        self.alpha = np.empty(self.num_envs, dtype=np.float64)
        self.bank = np.empty(self.num_envs, dtype=np.float64)
        self.wind_frame = np.empty(self.num_envs, dtype=np.float64)
        self.vario = np.empty(self.num_envs, dtype=np.float64)
        self.roll_cue = np.empty(self.num_envs, dtype=np.float64)
        self.initial_height = np.empty(self.num_envs, dtype=np.float64)
        self.initial_energy_height = np.empty(self.num_envs, dtype=np.float64)
        self._last_energy_height = np.empty(self.num_envs, dtype=np.float64)
        self._done = np.zeros(self.num_envs, dtype=bool)
        self._reset_slots(np.arange(self.num_envs), self._normalized_options(options))
        return self._observations(), self._infos(np.zeros(self.num_envs), np.full(self.num_envs, None, dtype=object))

    def step(self, actions, active_mask=None):
        actions = np.asarray(actions, dtype=np.float64)
        if actions.shape != (self.num_envs, 2):
            raise ValueError("batched dynamic actions must have shape (num_envs, 2)")
        if active_mask is None:
            active = np.ones(self.num_envs, dtype=bool)
        else:
            active = np.asarray(active_mask, dtype=bool)
            if active.shape != (self.num_envs,):
                raise ValueError("active_mask must have shape (num_envs,)")
        active &= ~self._done
        actions = np.clip(actions, 0.0, 1.0)
        termination_reasons = np.full(self.num_envs, None, dtype=object)
        terminated = np.zeros(self.num_envs, dtype=bool)
        truncated = np.zeros(self.num_envs, dtype=bool)
        rewards = np.zeros(self.num_envs, dtype=np.float64)
        if np.any(active):
            wind = self._wind(self.position[active], self.wind_frame[active])
            tas = np.linalg.norm(self.ground_velocity[active] - wind, axis=1)
            starting_energy_height = np.zeros(self.num_envs, dtype=np.float64)
            starting_energy_height[active] = self.position[active, 2] + 0.5 * tas * tas / self.physics.g
            roll_cue_sum = np.zeros(self.num_envs, dtype=np.float64)
            alive = active.copy()
            filter_gain = 1.0 - np.exp(-self.dt_integration / config.DYNAMIC_VARIO_TIME_CONSTANT)
            for _ in range(self.n_integration_steps):
                candidate, valid = self._rk4_candidates(actions, alive)
                divergence = alive & ~valid
                terminated[divergence] = True
                termination_reasons[divergence] = "numerical_divergence"
                alive &= valid
                self._set_states(candidate, alive)
                if not np.any(alive):
                    break
                selected = np.flatnonzero(alive)
                wind = self._wind(self.position[selected], self.wind_frame[selected])
                new_tas = np.linalg.norm(self.ground_velocity[selected] - wind, axis=1)
                energy_height = self.position[selected, 2] + 0.5 * new_tas * new_tas / self.physics.g
                raw_vario = (energy_height - starting_energy_height[selected]) / self.dt_integration
                self.vario[selected] += filter_gain * (raw_vario - self.vario[selected])
                acceleration, _, right, lift_direction, aerodynamic = self._aerodynamic_acceleration(
                    wind, self.ground_velocity[selected], self.alpha[selected], self.bank[selected]
                )
                roll_selected = selected[aerodynamic]
                if len(roll_selected):
                    half_span = config.WINGSPAN / 2.0
                    right_tip = self.position[roll_selected] + half_span * right[aerodynamic]
                    left_tip = self.position[roll_selected] - half_span * right[aerodynamic]
                    right_tip[:, 2] = np.clip(right_tip[:, 2], 0.0, self.domain_size[2] - 0.01)
                    left_tip[:, 2] = np.clip(left_tip[:, 2], 0.0, self.domain_size[2] - 0.01)
                    right_wind = self._wind(right_tip, self.wind_frame[roll_selected])
                    left_wind = self._wind(left_tip, self.wind_frame[roll_selected])
                    roll_cue_sum[roll_selected] += np.einsum(
                        "ij,ij->i", right_wind - left_wind, lift_direction[aerodynamic]
                    )
                starting_energy_height[selected] = energy_height
                low_altitude = alive & (self.position[:, 2] <= self.domain_size[2] * config.DYNAMIC_ALTITUDE_MIN_FRACTION)
                high_altitude = alive & (self.position[:, 2] >= self.domain_size[2] * config.DYNAMIC_ALTITUDE_MAX_FRACTION)
                wind_end = alive & (self.wind_frame >= self.wind_manager.max_t_idx)
                terminated[low_altitude] = True
                termination_reasons[low_altitude] = "altitude_low"
                terminated[high_altitude] = True
                termination_reasons[high_altitude] = "altitude_high"
                truncated[wind_end & ~terminated] = True
                termination_reasons[wind_end & ~terminated] = "wind_end"
                alive &= ~(terminated | truncated)

            self.roll_cue[active] = roll_cue_sum[active] / self.n_integration_steps
            ending_wind = self._wind(self.position[active], self.wind_frame[active])
            ending_tas = np.linalg.norm(self.ground_velocity[active] - ending_wind, axis=1)
            ending_energy_height = self.position[active, 2] + 0.5 * ending_tas * ending_tas / self.physics.g
            active_reasons = termination_reasons[active]
            rewards[active] = ending_energy_height - self._last_energy_height[active]
            active_indices = np.flatnonzero(active)
            rewards[active_indices[active_reasons == "numerical_divergence"]] = 0.0
            self._last_energy_height[active] = ending_energy_height

        infos = self._infos(rewards, termination_reasons)
        finished = terminated | truncated
        if self.autoreset and np.any(finished):
            self._reset_slots(np.flatnonzero(finished), [None] * int(finished.sum()))
        else:
            self._done |= finished
        return self._observations(), rewards.astype(np.float32), terminated, truncated, infos

    @property
    def unwrapped(self):
        return self

    def close(self):
        if self._owns_wind_manager:
            self.wind_manager.close()


def dynamic_discrete_action_commands(actions, action_levels=config.DYNAMIC_DQN_ACTION_LEVELS):
    actions = np.asarray(actions)
    if action_levels < 2:
        raise ValueError("dynamic DQN requires at least two action levels")
    if np.any(actions < 0) or np.any(actions >= action_levels * action_levels):
        raise ValueError("dynamic discrete actions are outside the configured action space")
    action_indices = actions.astype(np.int64)
    speed_indices, roll_indices = np.divmod(action_indices, action_levels)
    return np.column_stack(
        [speed_indices / (action_levels - 1), roll_indices / (action_levels - 1)]
    ).astype(np.float32)


class DynamicDiscreteActionBatchWrapper:
    """Expose discrete DQN actions over a DynamicGliderBatchEnv."""

    def __init__(self, env, action_levels=config.DYNAMIC_DQN_ACTION_LEVELS):
        self.env = env
        self.num_envs = env.num_envs
        self.action_levels = action_levels
        self.single_action_space = spaces.Discrete(action_levels * action_levels)
        self.single_observation_space = env.single_observation_space
        self.action_space = self.single_action_space
        self.observation_space = self.single_observation_space

    def reset(self, seed=None, options=None):
        return self.env.reset(seed=seed, options=options)

    def step(self, actions, active_mask=None):
        actions = np.asarray(actions)
        if actions.shape != (self.num_envs,):
            raise ValueError("batched dynamic discrete actions must have shape (num_envs,)")
        return self.env.step(
            dynamic_discrete_action_commands(actions, self.action_levels),
            active_mask=active_mask,
        )

    @property
    def wind_manager(self):
        return self.env.wind_manager

    @property
    def unwrapped(self):
        return self.env

    def close(self):
        self.env.close()


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
        return dynamic_discrete_action_commands(np.asarray([action]), self.action_levels)[0]

    def command_to_action(self, command):
        command = np.asarray(command, dtype=np.float32)
        if command.shape != (2,) or np.any(command < 0.0) or np.any(command > 1.0):
            raise ValueError("dynamic command must have shape (2,) and values in [0, 1]")
        indices = np.rint(command * (self.action_levels - 1)).astype(int)
        return int(indices[0] * self.action_levels + indices[1])
