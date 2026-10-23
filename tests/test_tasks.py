"""Native task compatibility, safe rewards and browser/native parity."""
import atexit
import json
import math
from pathlib import Path
import shutil
import subprocess

from fastapi.testclient import TestClient
import numpy as np
from pydantic import ValidationError
import pytest

from morpholab.environment import WalkerEnv
from morpholab.server import create_app
from morpholab.tasks import TaskConfig, compile_reward, evaluate_task
from morpholab.training import TrainConfig, evaluate_policy


STATE = {
    "position": [1.1, .2, .8], "velocity": [.7, -.1, .05], "height": .8,
    "initial_height": .9, "upright": .95, "target_speed": 1.2,
    "energy": .4, "smoothness": .2, "elapsed": 3, "previous_distance": 1.1, "dt": .02,
}


def javascript(source, payload):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is required to compare browser task semantics")
    module = (Path(__file__).resolve().parents[1] / "morpholab/static/task-regimes.mjs").as_uri()
    script = (
        f"import {{validateTask,evaluateTask,compileReward}} from {json.dumps(module)};"
        "let input='';for await(const chunk of process.stdin)input+=chunk;"
        "const data=JSON.parse(input);"
        + source
    )
    result = subprocess.run(
        [node, "--input-type=module", "-e", script], input=json.dumps(payload),
        capture_output=True, text=True, check=True,
    )
    return json.loads(result.stdout)


@pytest.mark.parametrize("preset", ["velocity", "balance", "target", "custom"])
def test_native_reward_matches_browser(preset):
    config = {
        "preset": preset, "target": [2, 0, .8],
        "expression": "target_distance < -0.5 ? speed_tracking + 0.2*upright : target_success",
    }
    native = evaluate_task(TaskConfig(**config), STATE)
    browser = javascript(
        "console.log(JSON.stringify(evaluateTask(validateTask(data.config),data.state)));",
        {"config": config, "state": STATE},
    )
    assert native["reward"] == pytest.approx(browser["reward"])
    assert native["terminated"] == browser["terminated"]
    assert native["terms"] == pytest.approx(browser["terms"])
    assert native["components"] == pytest.approx(browser["components"])


@pytest.mark.parametrize("expression", [
    "-2^2 + 2**3**2", "x>0 && y<1 ? max(x,y,3) : 0",
    "!x || y != 0", "clamp(x,-1,1) + abs(y)", "-5 % 3",
    "x ? y ? 3 : 4 : 5", "sqrt(4)+exp(0)+log(1)+sin(pi/2)+cos(0)+tanh(0)",
    "0 && 1/0", "1 || 1/0",
])
def test_expression_operator_semantics_match_browser(expression):
    variables = {"x": 2, "y": -.5, "pi": math.pi}
    expected = javascript(
        "console.log(JSON.stringify(compileReward(data.expression)(data.variables)));",
        {"expression": expression, "variables": variables},
    )
    assert compile_reward(expression)(variables) == pytest.approx(expected)


@pytest.mark.parametrize("source", [
    "__import__('os').system('id')", "globalThis.process", "x.constructor", "[1,2][0]",
    "unknown + 1", "eval(x)", "1e999", "1; 2", "(" * 70 + "1" + ")" * 70,
    "+".join(["x"] * 260),
])
def test_reward_language_rejects_code_and_excessive_complexity(source):
    with pytest.raises(ValueError):
        compile_reward(source)


@pytest.mark.parametrize("source", ["1/0", "sqrt(-1)", "exp(10000)", "pow(1)"])
def test_invalid_custom_reward_result_is_reported(source):
    with pytest.raises(ValueError, match="Custom reward"):
        compile_reward(source)({})


@pytest.mark.parametrize("task", [
    {"episode_seconds": 0}, {"episode_seconds": 121}, {"target": [0, 0]},
    {"target": [1001, 0, 0]}, {"target": [True, 0, 0]}, {"terminate_on_fall": "false"},
    {"weights": {"velocity": 1}}, {"weights": {"alive": 101}},
    {"preset": "custom", "expression": "open(x)"}, {"unknown": 1},
])
def test_task_config_rejects_invalid_settings(task):
    with pytest.raises(ValidationError):
        TaskConfig.model_validate(task)


def test_explicit_tasks_append_nine_observations_without_changing_legacy_prefix():
    legacy, task = WalkerEnv(), WalkerEnv(task={"preset": "target", "target": [2, 1, .7]})
    try:
        old, _ = legacy.reset(seed=17)
        new, _ = task.reset(seed=17)
        assert new.shape[0] == old.shape[0] + 9
        np.testing.assert_array_equal(new[:len(old)], old)
        np.testing.assert_allclose(new[-9:-6], task.data.xpos[task._torso_id], atol=1e-7)
        np.testing.assert_allclose(new[-6:-3], np.array([2, 1, .7]) - task.data.xpos[task._torso_id], atol=1e-7)
        np.testing.assert_array_equal(new[-3:], 0)
        assert task.get_frame()["target"] == [2, 1, .7]
        assert task.observation_schema == "rlsim-planar-task-v1"
        assert TrainConfig().task is None
    finally:
        legacy.close()
        task.close()


def test_episode_duration_and_disabled_fall_termination_are_honored():
    config = TrainConfig(task=TaskConfig(episode_seconds=1, terminate_on_fall=False))
    result = evaluate_policy(config, capture=True)
    assert result["summary"]["steps"] == 50
    assert result["summary"]["seconds"] == pytest.approx(1)
    assert result["task"]["episode_seconds"] == 1
    assert result["model_info"]["observation_size"] == 43


def test_tracked_body_names_are_resolved_and_absent_names_rejected():
    env = WalkerEnv(task={"tracked_body": "lower_0"})
    try:
        assert env._tracked_id != env._torso_id
        np.testing.assert_allclose(env._task_position, env.data.body("lower_0").xpos)
    finally:
        env.close()
    with pytest.raises(ValueError, match="Tracked body"):
        WalkerEnv(task={"tracked_body": "missing"})


@pytest.mark.parametrize("task", [None, {"tracked_body": "lower_0"}])
def test_controller_interface_metadata_describes_actual_observation_and_action_order(task):
    env = WalkerEnv(task=task)
    try:
        info = env.model_info()
        assert len(info["observation_names"]) == info["observation_size"]
        assert len(info["action_names"]) == info["action_size"]
        assert info["action_names"] == [actuator["name"] for actuator in info["actuators"]]
        assert all(actuator["normalized_range"] == [-1, 1] for actuator in info["actuators"])
        np.testing.assert_array_equal(
            [actuator["control_range"] for actuator in info["actuators"]],
            env.model.actuator_ctrlrange,
        )
        assert info["timestep"] == env.dt
        assert info["tracked_body"] == ("lower_0" if task else "torso")
        assert set(joint["type"] for joint in info["joints"]) == {"slide", "hinge"}
        assert info["observation_names"][0] == "qpos[1]"
        if task:
            assert info["observation_names"][-3:] == [
                "body_velocity.x", "body_velocity.y", "body_velocity.z",
            ]
    finally:
        env.close()


def test_native_task_preview_and_untrained_replay_api(tmp_path):
    app = create_app(tmp_path / "runs")
    try:
        with TestClient(app) as client:
            preview = client.post("/api/preview", json={
                "body": {}, "task": {"preset": "target", "target": [3, 1, .7]},
            })
            assert preview.status_code == 200, preview.text
            assert preview.json()["frame"]["target"] == [3, 1, .7]
            assert preview.json()["observation_size"] == 43
            assert any(body["name"] == "torso" for body in preview.json()["model_info"]["bodies"])
            replay = client.post("/api/replay", json={
                "body": {}, "task": {"episode_seconds": 1, "terminate_on_fall": False},
            })
            assert replay.status_code == 200, replay.text
            assert replay.json()["summary"]["steps"] == 50
            for endpoint in ("/api/preview", "/api/replay", "/api/train"):
                result = client.post(endpoint, json={"body": {}, "task": {"tracked_body": "missing"}})
                assert result.status_code == 422, result.text
            assert not list((tmp_path / "runs").iterdir())
    finally:
        atexit.unregister(app.state.manager.close)
