"""Configuration must reach real PPO training and survive checkpoint restore."""

import atexit
import sys
from unittest.mock import Mock

from fastapi.testclient import TestClient
import numpy as np
from pydantic import ValidationError
import pytest
import torch

from morpholab import cli, server, training


DEFAULTS = {
    "hidden_size": 256,
    "hidden_layers": 2,
    "learning_rate": 0.0003,
    "batch_size": 512,
    "epochs": 10,
    "rollout_steps": 256,
    "gamma": 0.99,
    "entropy_coef": 0.005,
    "clip_range": 0.2,
}
CUSTOM = {
    "hidden_size": 16,
    "hidden_layers": 3,
    "learning_rate": 0.0006,
    "batch_size": 2048,
    "epochs": 1,
    "rollout_steps": 64,
    "gamma": 0.92,
    "entropy_coef": 0.02,
    "clip_range": 0.15,
}


def test_legacy_config_keeps_original_native_network_and_optimizer_defaults():
    config = training.TrainConfig.model_validate({"total_steps": 4097, "n_envs": 8})
    assert {key: getattr(config, key) for key in DEFAULTS} == DEFAULTS
    assert config.actual_steps == 6144


@pytest.mark.parametrize(
    "field,invalid",
    [
        ("hidden_size", 15), ("hidden_size", 257),
        ("hidden_layers", 0), ("hidden_layers", 5),
        ("learning_rate", 0.000009), ("learning_rate", 0.01001),
        ("batch_size", 15), ("batch_size", 2049),
        ("epochs", 0), ("epochs", 21),
        ("rollout_steps", 15), ("rollout_steps", 1025),
        ("gamma", 0.7999), ("gamma", 1.0),
        ("entropy_coef", -0.0001), ("entropy_coef", 0.10001),
        ("clip_range", 0.0499), ("clip_range", 0.4001),
        ("learning_rate", float("nan")), ("gamma", float("inf")),
        ("hidden_layers", 2.5),
    ],
)
def test_invalid_network_and_optimizer_values_are_rejected(field, invalid):
    with pytest.raises(ValidationError):
        training.TrainConfig(**{field: invalid})


def test_budget_rounding_uses_configured_rollout_and_environment_count():
    config = training.TrainConfig(total_steps=4097, n_envs=3, rollout_steps=64)
    assert config.actual_steps == 4224
    exact = training.TrainConfig(total_steps=4096, n_envs=1, rollout_steps=16)
    assert exact.actual_steps == 4096


def test_api_exposes_defaults_and_persists_requested_settings(tmp_path, monkeypatch):
    hardware = {"training_device": "cpu", "execution_location": "server"}
    monkeypatch.setattr(server, "cached_hardware", lambda: hardware)
    monkeypatch.setattr(server, "hardware_info", lambda _: hardware)
    worker = Mock()
    worker.poll.return_value = None
    launch = Mock(return_value=worker)
    monkeypatch.setattr(server.subprocess, "Popen", launch)
    app = server.create_app(tmp_path)
    try:
        with TestClient(app) as client:
            defaults = client.get("/api/config").json()["defaults"]
            assert {key: defaults[key] for key in DEFAULTS} == DEFAULTS
            invalid = client.post("/api/train", json={"hidden_size": 500})
            assert invalid.status_code == 422
            launch.assert_not_called()
            result = client.post("/api/train", json={
                **CUSTOM, "n_envs": 3, "total_steps": 4097, "device": "cpu",
            })
            assert result.status_code == 200, result.text
            state = result.json()
            assert state["total_steps"] == 4224
            config = training.read_json(tmp_path / state["run_id"] / "config.json")
            assert {key: config[key] for key in CUSTOM} == CUSTOM
            assert {key: state["config"][key] for key in CUSTOM} == CUSTOM
            launch.assert_called_once()
    finally:
        atexit.unregister(app.state.manager.close)
        app.state.manager.process = None


def test_cli_forwards_network_and_optimizer_flags(tmp_path, monkeypatch, capsys):
    manager = Mock()
    manager.start.return_value = {"run_id": "scratch"}
    manager.resolve.return_value = tmp_path
    manager.status.return_value = {"status": "completed"}
    monkeypatch.setattr(server, "RunManager", lambda _: manager)
    arguments = ["rlsim", "train", "--steps", "4096", "--device", "cpu"]
    for key, value in CUSTOM.items():
        arguments.extend(["--" + key.replace("_", "-"), str(value)])
    monkeypatch.setattr(sys, "argv", arguments)
    cli.main()
    config = manager.start.call_args.args[0]
    assert {key: getattr(config, key) for key in CUSTOM} == CUSTOM
    assert config.device == "cpu"
    manager.close.assert_called_once()
    capsys.readouterr()


def test_custom_network_trains_on_cpu_and_restores_with_matching_normalization(tmp_path):
    config = training.TrainConfig(
        **CUSTOM, total_steps=4096, n_envs=2, device="cpu",
        curriculum=False, randomize=False,
    )
    training.atomic_json(tmp_path / "config.json", config.model_dump())
    training.atomic_json(tmp_path / "status.json", {"status": "starting", "history": []})
    training.train_worker(tmp_path)
    status = training.read_json(tmp_path / "status.json")
    assert status["status"] == "completed", status.get("error")
    assert status["steps"] == config.actual_steps
    assert status["policy_device"] == "cpu"
    restored_config, model, normalizer = training.load_checkpoint(tmp_path)
    try:
        assert restored_config == config
        assert model._n_updates > 0
        assert model.num_timesteps == 4096
        assert model.n_steps == config.rollout_steps
        assert model.batch_size == config.n_envs * config.rollout_steps
        assert model.n_epochs == config.epochs
        assert model.learning_rate == config.learning_rate
        assert model.gamma == config.gamma
        assert normalizer.gamma == config.gamma
        assert model.ent_coef == config.entropy_coef
        assert model.clip_range(1) == config.clip_range
        expected_layers = [config.hidden_size] * config.hidden_layers
        for network in [
            model.policy.mlp_extractor.policy_net,
            model.policy.mlp_extractor.value_net,
        ]:
            assert [
                layer.out_features for layer in network if isinstance(layer, torch.nn.Linear)
            ] == expected_layers
        action, _ = model.predict(normalizer.reset(), deterministic=True)
        assert np.isfinite(action).all()
        assert action.shape == (1, config.body.legs * 2)
    finally:
        normalizer.close()
