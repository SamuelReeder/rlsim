"""Exercise the browser/API boundary without launching expensive training jobs."""

import atexit
import io
import json
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock
import zipfile

from fastapi.testclient import TestClient
import pytest

from morpholab.config import EnvironmentConfig, PRESETS
from morpholab import server, training


CPU_ONLY = {
    "cuda_available": False,
    "gpu_name": None,
    "torch_version": "test",
    "cuda_version": None,
    "physics_device": "cpu",
    "training_device": "cpu",
    "mps_available": False,
    "requested_device": "auto",
    "fallback_reason": "No supported GPU available",
    "execution_location": "server",
}


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "cached_hardware", lambda: dict(CPU_ONLY))
    monkeypatch.setattr(
        server, "hardware_info",
        lambda device="auto": {**CPU_ONLY, "requested_device": device},
    )
    app = server.create_app(tmp_path / "runs")
    with TestClient(app) as test_client:
        yield test_client
    # Test workers are mocks. Keep atexit from attempting to wait for them.
    atexit.unregister(app.state.manager.close)
    app.state.manager.process = None


@pytest.fixture
def worker(monkeypatch):
    process = Mock()
    process.poll.return_value = None
    launch = Mock(return_value=process)
    monkeypatch.setattr(server, "subprocess", SimpleNamespace(
        Popen=launch, TimeoutExpired=subprocess.TimeoutExpired, STDOUT=subprocess.STDOUT,
    ))
    return launch, process


def write_completed_run(client):
    manager = client.app.state.manager
    run_id = "20261020T140000-0123abcd"
    path = manager.root / run_id
    path.mkdir()
    config = training.TrainConfig(total_steps=4096).model_dump()
    training.atomic_json(path / "config.json", config)
    training.atomic_json(
        path / "status.json",
        {
            "run_id": run_id,
            "status": "completed",
            "steps": 4096,
            "created_at": "2026-10-20T14:00:00+00:00",
            "config": config,
            "checkpoint_available": True,
        },
    )
    return path


@pytest.mark.parametrize("body", PRESETS.values(), ids=PRESETS.keys())
def test_preview_returns_real_geometry_and_policy_dimensions(client, body):
    response = client.post("/api/preview", json=body.model_dump())
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["body"] == body.model_dump()
    assert result["environment"] == EnvironmentConfig().model_dump()
    assert result["action_size"] == 2 * body.legs
    assert result["observation_size"] > result["action_size"]
    assert result["frame"]["time"] == 0
    assert result["frame"]["torso_position"][2] > 0
    assert any(geom["type"] == "plane" for geom in result["frame"]["geoms"])
    assert len(result["frame"]["geoms"]) > body.legs * 2
    json.dumps(result, allow_nan=False)
    assert client.get("/api/status").json()["status"] == "idle"


@pytest.mark.parametrize(
    "endpoint,payload",
    [
        ("/api/preview", {"legs": 3}),
        ("/api/preview", {"upper_length": 0}),
        ("/api/preview", {"motor_strength": 151}),
        ("/api/preview", {"name": ""}),
        ("/api/preview", {"unknown": True}),
        ("/api/train", {"total_steps": 100}),
        ("/api/train", {"total_steps": 2_000_001}),
        ("/api/train", {"target_speed": -1}),
        ("/api/train", {"n_envs": 33}),
        ("/api/train", {"body": {"legs": 7}}),
        ("/api/train", {"device": "webgpu"}),
        ("/api/preview", {"body": {}, "environment": {"gravity": 0}}),
        ("/api/preview", {"body": {}, "environment": {"friction": 4}}),
        ("/api/train", {"environment": {"gravity": 21}}),
        ("/api/train", {"environment": {"friction": 0}}),
        ("/api/replay", {"environment": {"gravity": 0}}),
        ("/api/replay", {"target_speed": 0}),
        ("/api/replay", {"seed": -1}),
    ],
)
def test_invalid_requests_are_rejected_without_creating_a_run(client, endpoint, payload):
    response = client.post(endpoint, json=payload)
    assert response.status_code == 422, response.text
    assert list(client.app.state.manager.root.iterdir()) == []


@pytest.mark.parametrize(
    "run_id",
    ["../outside", "/tmp/outside", "20261020T140000-0123abcd/../../secret", "bad-id"],
)
def test_run_ids_cannot_address_other_paths(client, run_id):
    responses = [
        client.post("/api/replay", json={"run_id": run_id}),
        client.get("/api/checkpoint", params={"run_id": run_id}),
        client.get("/api/body", params={"run_id": run_id}),
    ]
    assert [response.status_code for response in responses] == [400, 400, 400]


def test_valid_but_missing_run_is_not_found(client):
    run_id = "20261020T140000-0123abcd"
    assert client.post("/api/replay", json={"run_id": run_id}).status_code == 404
    assert client.get("/api/checkpoint", params={"run_id": run_id}).status_code == 404
    assert client.get("/api/body", params={"run_id": run_id}).status_code == 404


@pytest.mark.parametrize("device", ["auto", "cpu"])
def test_cpu_only_server_accepts_training(client, worker, monkeypatch, device):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(server, "cached_hardware", training.hardware_info)
    monkeypatch.setattr(server, "hardware_info", training.hardware_info)
    launch, _ = worker
    hardware = client.get("/api/config").json()["hardware"]
    assert hardware["cuda_available"] is False
    assert hardware["training_device"] == "cpu"
    assert hardware["physics_device"] == "cpu"
    assert hardware["execution_location"] == "server"
    response = client.post("/api/train", json={"total_steps": 4096, "device": device})
    assert response.status_code == 200, response.text
    state = response.json()
    assert state["hardware"]["training_device"] == "cpu"
    assert state["config"]["device"] == device
    launch.assert_called_once()


@pytest.mark.parametrize("device", ["cuda", "mps"])
def test_explicit_unavailable_device_never_launches_training(client, monkeypatch, device):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(server, "hardware_info", training.hardware_info)
    launch = Mock()
    monkeypatch.setattr(server, "subprocess", SimpleNamespace(
        Popen=launch, TimeoutExpired=subprocess.TimeoutExpired, STDOUT=subprocess.STDOUT,
    ))
    response = client.post("/api/train", json={"total_steps": 4096, "device": device})
    assert response.status_code == 409, response.text
    assert f"Cannot train on {device.upper()}" in response.json()["detail"]
    assert "device='auto'" in response.json()["detail"]
    launch.assert_not_called()
    assert list(client.app.state.manager.root.iterdir()) == []


def test_active_run_rejects_overlapping_training_replay_and_export(client, worker):
    launch, process = worker
    response = client.post("/api/train", json={"total_steps": 4097, "n_envs": 8})
    assert response.status_code == 200, response.text
    state = response.json()
    assert state["status"] == "starting"
    assert state["total_steps"] == 6144
    assert client.get("/api/status").json()["run_id"] == state["run_id"]
    assert client.post("/api/train", json={"total_steps": 4096}).status_code == 409
    assert client.post("/api/replay", json={"run_id": state["run_id"]}).status_code == 409
    assert client.get("/api/checkpoint", params={"run_id": state["run_id"]}).status_code == 409
    launch.assert_called_once()
    command = launch.call_args.args[0]
    assert command[1:4] == ["-m", "morpholab.cli", "worker"]
    assert command[-1] == str(client.app.state.manager.root / state["run_id"])
    assert client.get("/api/body", params={"run_id": state["run_id"]}).status_code == 200
    stopped = client.post("/api/stop")
    assert stopped.json()["status"] == "stopping"
    assert (client.app.state.manager.root / state["run_id"] / "STOP").is_file()
    assert client.get("/api/status").json()["status"] == "stopping"
    process.poll.return_value = 1
    failed = client.get("/api/status").json()
    assert failed["status"] == "failed"
    assert "Worker exited" in failed["error"]


def test_failed_launch_is_reported_and_persisted(client, worker):
    launch, _ = worker
    launch.side_effect = OSError("process launch failed")
    response = client.post("/api/train", json={"total_steps": 4096})
    assert response.status_code == 500
    state = client.get("/api/status").json()
    assert state["status"] == "failed"
    assert "process launch failed" in state["error"]


def test_checkpoint_download_contains_policy_normalizer_and_body(client):
    path = write_completed_run(client)
    (path / "policy.zip").write_bytes(b"test-policy")
    (path / "normalize.pkl").write_bytes(b"test-normalizer")
    (path / "body.xml").write_text("<mujoco/>", encoding="utf-8")
    response = client.get("/api/checkpoint", params={"run_id": path.name})
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        assert set(archive.namelist()) == {
            "policy.zip", "normalize.pkl", "config.json", "body.xml", "status.json"
        }
        assert archive.read("normalize.pkl") == b"test-normalizer"
        assert json.loads(archive.read("config.json"))["body"]["legs"] == 4


def test_saved_run_without_policy_rejects_replay_and_download(client):
    path = write_completed_run(client)
    assert client.post("/api/replay", json={"run_id": path.name}).status_code == 409
    assert client.get("/api/checkpoint", params={"run_id": path.name}).status_code == 404


def test_continuation_rejects_legacy_models_and_invalid_budget(client):
    path = write_completed_run(client)
    response = client.post('/api/resume', json={'run_id': path.name, 'additional_steps':4096})
    assert response.status_code == 409
    assert 'continuation' in response.json()['detail']
    assert client.post('/api/resume',json={'run_id':path.name,'additional_steps':0}).status_code==422
    assert client.post('/api/resume',json={'run_id':'../other','additional_steps':4096}).status_code==400


def test_best_checkpoint_api_selection_library_and_download(client, monkeypatch):
    path = write_completed_run(client)
    for name in ["policy.zip", "normalize.pkl", "best_policy.zip", "best_normalize.pkl"]:
        (path / name).write_bytes(name.encode())
    best = {"steps": 2048, "validation": {"mean_return": 12, "survival_rate": 1, "speed_error": .1}}
    training.atomic_json(path / "best.json", best)
    state = training.read_json(path / "status.json")
    state.update(best_checkpoint_steps=best["steps"], best_validation=best["validation"])
    training.atomic_json(path / "status.json", state)
    training.atomic_json(path / "replay.json", {"source": "trained", "frames": [], "summary": {"seed": 10007}})
    assert client.get("/api/runs").json()["runs"][0]["best"] == best
    infer = Mock(return_value={"source": "fresh-best"})
    monkeypatch.setattr(server, "replay_run", infer)
    assert client.post("/api/replay", json={"run_id": path.name}).json()["source"] == "fresh-best"
    infer.assert_called_once_with(path, checkpoint="best")
    infer.reset_mock()
    assert client.post("/api/replay", json={"run_id": path.name, "checkpoint": "latest"}).status_code == 200
    infer.assert_not_called()
    with zipfile.ZipFile(io.BytesIO(client.get("/api/checkpoint", params={"run_id": path.name}).content)) as archive:
        assert {"policy.zip", "normalize.pkl", "best_policy.zip", "best_normalize.pkl", "best.json"} <= set(archive.namelist())


def test_untrained_replay_is_explicitly_labelled_and_runs_actual_physics(client):
    response = client.post("/api/replay", json={"body": PRESETS["quadruped"].model_dump()})
    assert response.status_code == 200, response.text
    replay = response.json()
    assert replay["source"] == "untrained"
    assert replay["summary"]["steps"] > 0
    assert replay["frames"][-1]["time"] > replay["frames"][0]["time"]
    assert replay["summary"]["mean_squared_action"] == 0
    json.dumps(replay, allow_nan=False)


@pytest.mark.parametrize("during", ["load", "prediction"])
def test_stop_inference_exits_generation_and_closes_resources(client, monkeypatch, during):
    import numpy as np
    from morpholab.tasks import TaskConfig
    path = write_completed_run(client)
    (path / "policy.zip").write_bytes(b"test")
    config = training.TrainConfig(task=TaskConfig(episode_seconds=120, terminate_on_fall=False))
    entered, release = threading.Event(), threading.Event()
    predictions = []
    normalizer = Mock()
    normalizer.normalize_obs.side_effect = lambda obs: obs

    def predict(obs, deterministic=True):
        predictions.append(1)
        if len(predictions) == 5:
            entered.set()
            assert release.wait(5)
        return np.zeros(config.body.legs * 2), None

    model = SimpleNamespace(predict=predict, num_timesteps=4096)

    def load(*args):
        if during == "load":
            entered.set()
            assert release.wait(5)
        return config, model, normalizer

    monkeypatch.setattr(training, "load_checkpoint", load)
    with ThreadPoolExecutor() as pool:
        future = pool.submit(client.post, "/api/replay", json={
            "run_id": path.name, "seed": 11, "replay_id": "cancel-this",
        })
        try:
            assert entered.wait(5)
            assert client.post("/api/stop-replay", json={"replay_id": "another-request"}).status_code == 200
            assert not client.app.state.replay_jobs.active["cancel-this"].is_set()
            assert client.post("/api/stop-replay", json={"replay_id": "cancel-this"}).status_code == 200
            count = len(predictions)
        finally:
            release.set()
        response = future.result(timeout=5)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["cancelled"] is True
    assert len(predictions) == count
    if during == "prediction":
        assert result["summary"]["steps"] == 4
        assert result["frames"][-1]["time"] == result["summary"]["seconds"]
    else:
        assert result["frames"] == []
    normalizer.close.assert_called_once()
    assert not client.app.state.replay_jobs.active
    json.dumps(result, allow_nan=False)


def test_stop_before_replay_starts_is_honored_and_request_can_be_replaced(client, monkeypatch):
    evaluate = Mock(return_value={"frames": [], "summary": {}})
    monkeypatch.setattr(server, "evaluate_policy", evaluate)
    assert client.post("/api/stop-replay", json={"replay_id": "early"}).status_code == 200
    payload = {"body": PRESETS["quadruped"].model_dump(), "replay_id": "early"}
    assert client.post("/api/replay", json=payload).json()["cancelled"] is True
    evaluate.assert_not_called()
    assert client.post("/api/replay", json={**payload, "replay_id": "fresh"}).status_code == 200
    evaluate.assert_called_once()
    assert not client.app.state.replay_jobs.active


def test_cancelled_before_first_step_has_finite_metrics():
    checks = iter([False, True])
    result = training.evaluate_policy(training.TrainConfig(), capture=True, cancel=lambda: next(checks))
    assert result["cancelled"] is True
    assert result["summary"]["steps"] == 0
    assert result["summary"]["mean_speed"] == 0
    json.dumps(result, allow_nan=False)


def test_environment_defaults_and_nested_preview(client):
    defaults = client.get("/api/config").json()
    assert defaults["environment_defaults"] == {"gravity": 9.81, "friction": 1.0}
    assert defaults["defaults"]["environment"] == defaults["environment_defaults"]
    response = client.post(
        "/api/preview",
        json={"body": PRESETS["biped"].model_dump(),
              "environment": {"gravity": 1.62, "friction": 0.2}},
    )
    assert response.status_code == 200, response.text
    assert response.json()["environment"] == {"gravity": 1.62, "friction": 0.2}
    assert response.json()["action_size"] == 4


def test_train_persists_environment_in_config_and_simulation(client, worker):
    import mujoco

    response = client.post(
        "/api/train",
        json={"total_steps": 4096, "environment": {"gravity": 3.71, "friction": 0.3}},
    )
    assert response.status_code == 200, response.text
    path = client.app.state.manager.root / response.json()["run_id"]
    saved = training.read_json(path / "config.json")
    assert saved["environment"] == {"gravity": 3.71, "friction": 0.3}
    model = mujoco.MjModel.from_xml_path(str(path / "body.xml"))
    assert model.opt.gravity[2] == pytest.approx(-3.71)
    assert model.geom("floor").friction[0] == pytest.approx(0.3)


def test_legacy_cached_policy_uses_its_saved_body_and_default_environment(client, monkeypatch):
    path = write_completed_run(client)
    (path / "policy.zip").write_bytes(b"test-policy")
    old_config = training.read_json(path / "config.json")
    old_config.pop("environment")
    old_config.pop("device")
    training.atomic_json(path / "config.json", old_config)
    training.atomic_json(path / "replay.json", {
        "source": "trained", "frames": [], "summary": {"seed": 10007},
        "frame_dt": 0.04, "body": old_config["body"],
    })
    infer = Mock()
    monkeypatch.setattr(server, "replay_run", infer)
    response = client.post(
        "/api/replay",
        json={"run_id": path.name, "body": PRESETS["hexapod"].model_dump()},
    )
    assert response.status_code == 200, response.text
    replay = response.json()
    assert replay["body"] == old_config["body"]
    assert replay["environment"] == EnvironmentConfig().model_dump()
    assert replay["config"]["environment"] == replay["environment"]
    assert replay["config"]["device"] == "auto"
    assert replay["target_speed"] == old_config["target_speed"]
    infer.assert_not_called()


@pytest.mark.parametrize(
    "overrides", [
        {"environment": {"gravity": 1.62, "friction": 0.2}},
        {"target_speed": 2.0}, {"seed": 0},
    ],
)
def test_every_inference_override_bypasses_cached_replay(client, monkeypatch, overrides):
    path = write_completed_run(client)
    (path / "policy.zip").write_bytes(b"test-policy")
    training.atomic_json(path / "replay.json", {"source": "cached"})
    infer = Mock(return_value={"source": "fresh"})
    monkeypatch.setattr(server, "replay_run", infer)
    response = client.post("/api/replay", json={"run_id": path.name, **overrides})
    assert response.status_code == 200, response.text
    assert response.json()["source"] == "fresh"
    expected = dict(overrides)
    if "environment" in expected:
        expected["environment"] = EnvironmentConfig(**expected["environment"])
    infer.assert_called_once_with(path, checkpoint="best", **expected)


def test_replay_run_applies_overrides_and_preserves_checkpoint_body(tmp_path, monkeypatch):
    config = training.TrainConfig(body=PRESETS["biped"])
    model, normalizer = Mock(), Mock()
    model.num_timesteps = 4096
    monkeypatch.setattr(training, "load_checkpoint", lambda _, checkpoint="latest": (config, model, normalizer))
    evaluate = Mock(return_value={"frames": [], "summary": {}})
    monkeypatch.setattr(training, "evaluate_policy", evaluate)
    training.replay_run(
        tmp_path, environment={"gravity": 1.62, "friction": 0.2},
        target_speed=2.0, seed=23,
    )
    actual_config, actual_model, actual_normalizer = evaluate.call_args.args
    assert actual_config.body == config.body
    assert actual_config.environment == EnvironmentConfig(gravity=1.62, friction=0.2)
    assert actual_config.target_speed == 2.0
    assert actual_config.seed == 23
    assert actual_model is model and actual_normalizer is normalizer
    assert evaluate.call_args.kwargs == {"capture": True, "seed": 23}
    assert config.environment == EnvironmentConfig()
    assert config.target_speed == 1.0
    normalizer.close.assert_called_once()


def test_zero_torque_inference_uses_requested_world_speed_and_seed(client):
    settings = {"environment": {"gravity": 1.62, "friction": 0.2},
                "target_speed": 2.0, "seed": 23}
    response = client.post(
        "/api/replay", json={"body": PRESETS["biped"].model_dump(), **settings}
    )
    assert response.status_code == 200, response.text
    replay = response.json()
    assert replay["environment"] == settings["environment"]
    assert replay["config"]["environment"] == settings["environment"]
    assert replay["target_speed"] == settings["target_speed"]
    assert replay["seed"] == replay["summary"]["seed"] == settings["seed"]
    assert all(frame["target_speed"] == 2.0 for frame in replay["frames"])
    assert replay["summary"]["mean_squared_action"] == 0
    assert replay["summary"]["steps"] > 0
