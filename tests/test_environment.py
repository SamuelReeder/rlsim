"""Physics contracts that catch geometry, seeding and policy-shape regressions."""

import json

import mujoco
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from pydantic import ValidationError

from morpholab.config import BodyConfig, EnvironmentConfig, PRESETS
from morpholab.environment import WalkerEnv, export_mjcf


@pytest.mark.parametrize("body", PRESETS.values(), ids=PRESETS.keys())
def test_each_preset_compiles_and_rolls_out_finite(body):
    model = mujoco.MjModel.from_xml_string(export_mjcf(body))
    assert model.nu == body.legs * 2
    assert model.body_mass.sum() == pytest.approx(body.body_mass)
    env = WalkerEnv(body, seed=17)
    observation, info = env.reset(seed=17)
    assert env.observation_space.contains(observation)
    for _ in range(150):
        observation, reward, terminated, truncated, info = env.step(
            env.action_space.sample() * 0.25
        )
        assert env.observation_space.contains(observation)
        assert np.isfinite(observation).all()
        assert np.isfinite(reward)
        assert reward == pytest.approx(sum(info["reward_components"].values()))
        json.dumps(env.get_frame(), allow_nan=False)
        if terminated or truncated:
            observation, info = env.reset()
    env.close()


def test_seeded_randomization_and_rollout_are_reproducible():
    env = WalkerEnv(randomize=True)
    initial_a, info_a = env.reset(seed=314)
    actions = np.random.default_rng(8).uniform(-0.1, 0.1, size=(8, env.model.nu))
    states_a = [env.step(action)[0] for action in actions]
    initial_b, info_b = env.reset(seed=314)
    states_b = [env.step(action)[0] for action in actions]
    np.testing.assert_array_equal(initial_a, initial_b)
    np.testing.assert_array_equal(states_a, states_b)
    assert info_a == info_b
    assert info_a["mass_scale"] != 1.0
    assert info_a["motor_scale"] != 1.0
    assert info_a["friction_scale"] != 1.0


def test_dimensions_and_topology_change_the_simulation():
    small = WalkerEnv(BodyConfig(legs=2, torso_length=0.4, upper_length=0.2))
    large = WalkerEnv(BodyConfig(legs=6, torso_length=1.5, upper_length=0.8))
    assert small.action_space.shape == (4,)
    assert large.action_space.shape == (12,)
    assert small.observation_space.shape != large.observation_space.shape
    small_torso = small.model.geom("torso_geom").size
    large_torso = large.model.geom("torso_geom").size
    assert large_torso[1] > small_torso[1]
    assert large.get_frame()["torso_position"][2] > small.get_frame()["torso_position"][2]


def test_speed_curriculum_and_domain_parameters_are_observed():
    env = WalkerEnv(target_speed=2.0, randomize=True)
    env.set_training_progress(0.0)
    observation, info = env.reset(seed=3)
    target_index = env.model.nq - 1 + env.model.nv
    assert info["target_speed"] == pytest.approx(0.5)
    assert observation[target_index] == pytest.approx(0.5)
    assert np.max(np.abs(observation[-3:])) <= 1
    env.set_training_progress(1.0)
    assert env.target_speed == pytest.approx(2.0)


def test_gymnasium_contract():
    check_env(WalkerEnv(), skip_render_check=True)


def test_invalid_body_and_actions_are_rejected():
    with pytest.raises(ValidationError):
        BodyConfig(legs=3)
    with pytest.raises(ValidationError):
        BodyConfig(upper_length=float("nan"))
    env = WalkerEnv()
    with pytest.raises(ValueError):
        env.step(np.zeros(2))
    with pytest.raises(ValueError):
        env.step(np.full(env.model.nu, np.nan))


def test_gravity_changes_actual_free_fall_without_changing_policy_shape():
    earth = WalkerEnv(environment={"gravity": 9.81}, seed=12)
    moon = WalkerEnv(environment={"gravity": 1.62}, seed=12)
    assert earth.action_space == moon.action_space
    assert earth.observation_space == moon.observation_space
    for env in (earth, moon):
        # Lift the body clear of all contacts; free-fall velocity should
        # reflect the configured acceleration after one real control step.
        env.data.qpos[1] = 10.0
        env.data.qvel[:] = 0.0
        mujoco.mj_forward(env.model, env.data)
        env.step(np.zeros(env.action_space.shape))
        assert env.data.qvel[1] == pytest.approx(-env.environment.gravity * env.dt)
        env.close()
    assert abs(earth.data.qvel[1]) > abs(moon.data.qvel[1]) * 5


def test_configured_friction_is_used_by_contacts_and_randomization():
    env = WalkerEnv(environment={"friction": 0.2}, randomize=True)
    _, info = env.reset(seed=314)
    expected = 0.2 * info["friction_scale"]
    assert env.model.geom("floor").friction[0] == pytest.approx(expected)
    for _ in range(30):
        env.step(np.zeros(env.action_space.shape))
        if env.data.ncon:
            break
    assert env.data.ncon > 0
    assert env.data.contact[0].friction[0] == pytest.approx(expected)
    # Repeated resets scale the configured base, not the previous episode.
    env.reset(seed=314)
    assert env.model.geom("floor").friction[0] == pytest.approx(expected)
    env.close()


@pytest.mark.parametrize(
    "values",
    [{"gravity": 0}, {"gravity": 21}, {"friction": 0}, {"friction": 4},
     {"gravity": float("nan")}, {"friction": float("inf")}, {"terrain": "hills"}],
)
def test_invalid_environment_settings_are_rejected(values):
    with pytest.raises(ValidationError):
        EnvironmentConfig.model_validate(values)
