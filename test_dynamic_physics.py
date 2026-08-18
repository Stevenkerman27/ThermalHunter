import numpy as np

from glider_discrete_simp import RBWindField
from glider_dynamic import DynamicGliderBatchEnv


def test_vector_rms_uses_three_dimensional_speed_per_sample():
    wind_field = object.__new__(RBWindField)
    wind_field.dsets_list = [
        {
            "ux": np.array([1.0, 0.0]),
            "uy": np.array([2.0, 0.0]),
            "uz": np.array([2.0, 0.0]),
        }
    ]

    # The first sample has speed squared 1 + 4 + 4; the second is zero.
    assert np.isclose(wind_field.vector_rms(), np.sqrt(9.0 / 2.0))


def test_batch_low_tas_acceleration_keeps_gravity():
    env = object.__new__(DynamicGliderBatchEnv)
    env.gravity = np.array([0.0, 0.0, -9.81])
    wind = np.zeros((2, 3), dtype=np.float64)
    ground_velocity = np.zeros((2, 3), dtype=np.float64)
    alpha = np.zeros(2, dtype=np.float64)
    bank = np.zeros(2, dtype=np.float64)

    acceleration, tas, right, lift_direction, active = env._aerodynamic_acceleration(
        wind, ground_velocity, alpha, bank
    )

    assert np.all(tas < 5.0)
    assert not np.any(active)
    np.testing.assert_allclose(acceleration, np.tile(env.gravity, (2, 1)))
    np.testing.assert_allclose(right, np.zeros((2, 3)))
    np.testing.assert_allclose(lift_direction, np.zeros((2, 3)))
