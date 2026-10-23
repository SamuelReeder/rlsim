"""Device selection must establish usable kernels before starting PPO."""

from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest
import torch
from pydantic import ValidationError

from morpholab import training


@pytest.fixture
def device_probes(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "Test NVIDIA GPU")
    probe = Mock()
    monkeypatch.setattr(training, "_probe_device", probe)
    return probe


def test_auto_uses_cuda_when_its_training_kernels_work(monkeypatch, device_probes):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    hardware = training.hardware_info()
    assert hardware["training_device"] == "cuda"
    assert hardware["gpu_name"] == "Test NVIDIA GPU"
    assert hardware["fallback_reason"] is None
    device_probes.assert_called_once_with("cuda")


def test_auto_falls_back_to_cpu_after_cuda_kernel_failure(monkeypatch, device_probes):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def verify(device):
        if device == "cuda":
            raise RuntimeError("no kernel image is available")

    device_probes.side_effect = verify
    hardware = training.hardware_info()
    assert hardware["cuda_available"] is True
    assert hardware["training_device"] == "cpu"
    assert hardware["gpu_name"] is None
    assert "no kernel image" in hardware["fallback_reason"]
    assert device_probes.call_args_list == [call("cuda"), call("cpu")]


def test_auto_uses_mps_before_cpu(monkeypatch, device_probes):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    hardware = training.hardware_info()
    assert hardware["training_device"] == "mps"
    assert hardware["gpu_name"] == "Apple GPU"
    device_probes.assert_called_once_with("mps")


def test_auto_falls_back_if_device_discovery_raises(monkeypatch, device_probes):
    monkeypatch.setattr(torch.cuda, "is_available", Mock(side_effect=RuntimeError("driver error")))
    hardware = training.hardware_info()
    assert hardware["training_device"] == "cpu"
    assert "driver error" in hardware["fallback_reason"]
    device_probes.assert_called_once_with("cpu")


def test_explicit_cpu_does_not_probe_gpu(monkeypatch, device_probes):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    hardware = training.hardware_info("cpu")
    assert hardware["training_device"] == "cpu"
    assert hardware["requested_device"] == "cpu"
    assert hardware["fallback_reason"] is None
    device_probes.assert_called_once_with("cpu")


@pytest.mark.parametrize("device", ["cuda", "mps"])
def test_explicit_accelerator_failure_is_actionable_and_does_not_fallback(
    monkeypatch, device_probes, device,
):
    backend = torch.cuda if device == "cuda" else torch.backends.mps
    monkeypatch.setattr(backend, "is_available", lambda: True)
    device_probes.side_effect = RuntimeError("operation not supported")
    with pytest.raises(RuntimeError, match=f"Cannot train on {device.upper()}") as error:
        training.hardware_info(device)
    assert "device='auto'" in str(error.value)
    device_probes.assert_called_once_with(device)


def test_legacy_configs_default_to_auto():
    config = training.TrainConfig.model_validate({"total_steps": 4096})
    assert config.device == "auto"
    assert config.actual_steps == 4096
    with pytest.raises(ValidationError):
        training.TrainConfig(device="webgpu")
    with pytest.raises(ValueError, match="Device must be"):
        training.hardware_info("webgpu")


def test_real_cpu_forward_and_backward_probe():
    training._probe_device("cpu")


def test_worker_does_not_retry_or_reset_after_training_failure(tmp_path, monkeypatch):
    import stable_baselines3

    config = training.TrainConfig(total_steps=4096, device="cpu", n_envs=1)
    training.atomic_json(tmp_path / "config.json", config.model_dump())
    training.atomic_json(tmp_path / "status.json", {"status": "starting", "history": []})
    info = Mock(return_value={"training_device": "cpu", "execution_location": "server"})
    monkeypatch.setattr(training, "hardware_info", info)
    monkeypatch.setattr(training, "evaluate_suite", lambda _: {})
    policy = SimpleNamespace(parameters=lambda: iter([torch.nn.Parameter(torch.zeros(1))]))
    model = SimpleNamespace(
        policy=policy, device=torch.device("cpu"),
        learn=Mock(side_effect=RuntimeError("training operation failed")),
    )
    constructor = Mock(return_value=model)
    monkeypatch.setattr(stable_baselines3, "PPO", constructor)
    training.train_worker(tmp_path)
    state = training.read_json(tmp_path / "status.json")
    assert state["status"] == "failed"
    assert state["policy_device"] == "cpu"
    assert state["hardware"]["training_device"] == "cpu"
    assert "training operation failed" in state["error"]
    assert not (tmp_path / "policy.zip").exists()
    constructor.assert_called_once()
    assert constructor.call_args.kwargs["device"] == "cpu"
    info.assert_called_once_with("cpu")
    model.learn.assert_called_once()
