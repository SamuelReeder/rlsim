"""External PyTorch and native controllers must retain predictions in TF.js."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest
import torch
from torch import nn

from morpholab.environment import WalkerEnv
from morpholab.portable import export_checkpoint, export_torch_policy
from morpholab.training import TrainConfig, atomic_json


ROOT = Path(__file__).resolve().parents[1]


def mlp(input_size, output_size, width=16, depth=3, activation=nn.Tanh):
    modules = []
    for i in range(depth):
        modules.extend([nn.Linear(input_size if i == 0 else width, width), activation()])
    modules.append(nn.Linear(width, output_size))
    return nn.Sequential(*modules)


def tfjs_predictions(path, observations):
    if not shutil.which("node") or not (ROOT / "node_modules/@tensorflow/tfjs").exists():
        pytest.skip("TensorFlow.js parity requires Node and npm ci")
    program = """
      import fs from 'node:fs';
      import * as tf from '@tensorflow/tfjs';
      import {PPO, RunningNormalizer} from './morpholab/static/browser-policy.mjs';
      globalThis.tf = tf;
      await tf.setBackend('cpu');
      const {path,observations}=JSON.parse(fs.readFileSync(0,'utf8'));
      const record=JSON.parse(fs.readFileSync(path,'utf8'));
      const policy=await PPO.restore(record.policy);
      const normalizer=RunningNormalizer.restore(record.normalizer);
      try {
        const result=await policy.act(normalizer.normalize(observations),true);
        process.stdout.write(JSON.stringify({actions:result.actions,values:result.values}));
      } finally { policy.dispose(); }
    """
    result = subprocess.run(
        ["node", "--input-type=module", "-e", program], cwd=ROOT,
        input=json.dumps({"path": str(path), "observations": observations}),
        text=True, capture_output=True, check=True,
    )
    return json.loads(result.stdout)


def test_external_controller_weights_and_normalization_match_tensorflow(tmp_path):
    torch.manual_seed(15)
    actor, critic = mlp(34, 8), mlp(34, 1)
    mean = np.linspace(-0.3, 0.7, 34)
    variance = np.linspace(0.2, 1.7, 34)
    variance[0] = 0  # An invariant observation remains valid after normalization.
    observations = np.random.default_rng(1).normal(size=(5, 34))
    output = tmp_path / "custom.json"
    record = export_torch_policy(
        actor, critic, torch.full((8,), -0.7), TrainConfig(), output,
        observation_mean=mean, observation_variance=variance, observation_count=35,
        observation_clip=6, steps=123, name="External controller",
    )
    assert record["config"]["hidden_size"] == 16
    assert record["config"]["hidden_layers"] == 3
    assert record["config"]["body"]["name"] == "External controller"
    assert record["normalizer"]["m2"] == (variance * 35).tolist()
    assert record["policy"]["weights"]["actor_w1"]["shape"] == [34, 16]
    assert record["policy"]["weights"]["actor_w1"]["values"] == actor[0].weight.detach().T.reshape(-1).tolist()
    normalized = np.clip((observations - mean) / np.sqrt(variance + 1e-8), -6, 6).astype(np.float32)
    expected_actions = actor(torch.from_numpy(normalized)).detach().numpy()
    expected_values = critic(torch.from_numpy(normalized)).detach().numpy().ravel()
    actual = tfjs_predictions(output, observations.tolist())
    np.testing.assert_allclose(actual["actions"], expected_actions, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(actual["values"], expected_values, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    "change,error",
    [
        ("relu", "Tanh"), ("width", "width"), ("depth", "1 to 4"),
        ("unequal", "equal hidden widths"), ("input", "dimensions"),
        ("output", "dimensions"), ("critic", "same hidden width"),
        ("weight", "finite float32"), ("logstd", "finite"),
        ("variance", "nonnegative"), ("mean", "finite"), ("one_stat", "both"),
        ("count", "positive"), ("clip", "positive"), ("steps", "nonnegative"),
    ],
)
def test_invalid_controller_exports_leave_no_output(tmp_path, change, error):
    actor, critic = mlp(34, 8), mlp(34, 1)
    kwargs = {}
    log_std = torch.zeros(8)
    if change == "relu":
        actor = mlp(34, 8, activation=nn.ReLU)
    elif change == "width":
        actor = mlp(34, 8, width=8)
    elif change == "depth":
        actor = mlp(34, 8, depth=5)
    elif change == "unequal":
        actor[2] = nn.Linear(16, 32)
    elif change == "input":
        actor = mlp(33, 8)
    elif change == "output":
        actor = mlp(34, 7)
    elif change == "critic":
        critic = mlp(34, 1, width=32)
    elif change == "weight":
        with torch.no_grad():
            actor[0].weight[0, 0] = float("inf")
    elif change == "logstd":
        log_std[0] = float("nan")
    elif change == "variance":
        kwargs.update(observation_mean=np.zeros(34), observation_variance=np.full(34, -1))
    elif change == "mean":
        kwargs.update(observation_mean=np.full(34, np.nan), observation_variance=np.ones(34))
    elif change == "one_stat":
        kwargs.update(observation_mean=np.zeros(34))
    elif change == "count":
        kwargs.update(observation_count=0)
    elif change == "clip":
        kwargs.update(observation_clip=float("inf"))
    elif change == "steps":
        kwargs.update(steps=-1)
    output = tmp_path / "invalid.json"
    with pytest.raises(ValueError, match=error):
        export_torch_policy(actor, critic, log_std, TrainConfig(), output, **kwargs)
    assert not output.exists()


def test_untrained_template_has_zero_steps_and_explicit_identity_statistics(tmp_path):
    config = TrainConfig()
    record = export_torch_policy(
        mlp(34, 8), mlp(34, 1), torch.zeros(8), config, tmp_path / "template.json",
    )
    assert record["steps"] == 0
    assert record["normalizer"]["mean"] == [0] * 34
    assert record["normalizer"]["m2"] == [1] * 34
    assert record["normalizer"]["count"] == 1
    assert record["format"] == "morpholab-browser-run-v1"
    assert record["policy"]["format"] == "morpholab-browser-ppo-v1"


def test_native_checkpoint_infers_saved_architecture_and_matches_tensorflow(tmp_path):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    torch.set_num_threads(2)
    config = TrainConfig(device="cpu", total_steps=4096, n_envs=1)
    # This legacy config deliberately lacks network settings: the exporter must
    # discover the saved network instead of assuming the new default 256×256.
    atomic_json(tmp_path / "config.json", {
        key: config.model_dump()[key]
        for key in ("body", "environment", "total_steps", "n_envs", "device")
    })
    normalizer = VecNormalize(DummyVecEnv([lambda: WalkerEnv()]), gamma=0.91, clip_obs=7)
    try:
        model = PPO(
            "MlpPolicy", normalizer, device="cpu", seed=7, n_steps=16,
            batch_size=16, n_epochs=1, learning_rate=0.0005, gamma=0.91,
            clip_range=0.15, ent_coef=0.01,
            policy_kwargs={"net_arch": {"pi": [32], "vf": [32]}},
        )
        model.learn(total_timesteps=64)
        model.save(tmp_path / "policy.zip")
        normalizer.save(tmp_path / "normalize.pkl")
        observations = np.random.default_rng(2).normal(size=(5, 34)).astype(np.float32)
        normalized = normalizer.normalize_obs(observations.copy())
        expected, _ = model.predict(normalized, deterministic=True)
        output = tmp_path / "exported.json"
        record = export_checkpoint(tmp_path, output)
        assert record["config"]["hidden_size"] == 32
        assert record["config"]["hidden_layers"] == 1
        assert record["config"]["gamma"] == 0.91
        assert record["config"]["rollout_steps"] == 16
        assert record["steps"] == 64
        assert record["normalizer"]["clip"] == 7
        actual = tfjs_predictions(output, observations.tolist())
        np.testing.assert_allclose(np.clip(actual["actions"], -1, 1), expected, rtol=1e-5, atol=1e-6)
    finally:
        normalizer.close()
