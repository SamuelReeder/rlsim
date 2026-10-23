"""Export supported PyTorch controllers to RLsim's browser JSON format.

The portable controller has separate actor and critic MLPs, each containing
one to four equal-width Linear/Tanh hidden layers followed by a Linear output.
The actor outputs raw Gaussian means; MuJoCo clips actions to [-1, 1].
Files carry the existing ``morpholab-browser-*-v1`` identifiers for compatibility.
This is an inference export, not an optimizer-state checkpoint.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Mapping, Any
import uuid

import numpy as np
import torch
from torch import nn

from .environment import WalkerEnv, export_mjcf
from .training import TrainConfig, load_checkpoint


def _array(value, shape: tuple[int, ...], label: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().to(device="cpu", dtype=torch.float64).numpy()
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{label} must have shape {shape} and contain only finite numbers")
    return array


def _network(network: nn.Sequential, input_size: int, output_size: int, label: str):
    if not isinstance(network, nn.Sequential):
        raise ValueError(f"{label} must be a flat torch.nn.Sequential")
    modules = list(network.children())
    depth = (len(modules) - 1) // 2
    if not 1 <= depth <= 4 or len(modules) != 2 * depth + 1:
        raise ValueError(f"{label} needs 1 to 4 Linear/Tanh hidden layers and a Linear output")
    linears = modules[::2]
    if any(type(layer) is not nn.Linear for layer in linears):
        raise ValueError(f"{label} must alternate Linear and Tanh, ending with Linear")
    if any(type(layer) is not nn.Tanh for layer in modules[1::2]):
        raise ValueError(f"{label} hidden activations must be Tanh")
    width = linears[0].out_features
    if not 16 <= width <= 256:
        raise ValueError(f"{label} hidden width must be between 16 and 256")
    for index, layer in enumerate(linears):
        expected_input = input_size if index == 0 else width
        expected_output = output_size if index == depth else width
        if layer.in_features != expected_input or layer.out_features != expected_output:
            raise ValueError(f"{label} layer dimensions must match the body and use equal hidden widths")
    return linears, width, depth


def _weight(value, label: str) -> dict:
    if isinstance(value, torch.Tensor):
        value = value.detach().to(device="cpu", dtype=torch.float32).numpy()
    value = np.asarray(value, dtype=np.float32)
    if not np.isfinite(value).all():
        raise ValueError(f"{label} must contain finite float32 weights")
    return {"shape": list(value.shape), "values": value.reshape(-1).tolist()}


def export_torch_policy(
    actor: nn.Sequential,
    critic: nn.Sequential,
    log_std: torch.Tensor,
    config: TrainConfig | Mapping[str, Any],
    output: Path,
    *,
    observation_mean=None,
    observation_variance=None,
    observation_count: float = 1.0,
    observation_clip: float = 10.0,
    steps: int = 0,
    name: str | None = None,
) -> dict:
    """Write a browser-importable actor/critic and return the serialized record.

    Pass the observation normalization used during training. Omitting both mean
    and variance selects zero mean / unit variance. The browser applies
    ``clip((obs - mean) / sqrt(variance + 1e-8), -clip, clip)``.
    ``name`` optionally replaces the body's display name. Architecture metadata
    is inferred from the supplied networks, not trusted from ``config``.
    All parameters and statistics must be finite. Zero steps explicitly marks
    an untrained template; this function does not train or evaluate a controller.
    """
    config = TrainConfig.model_validate(
        config.model_dump() if isinstance(config, TrainConfig) else dict(config)
    )
    env = WalkerEnv(config.body, environment=config.environment, task=config.task)
    try:
        obs_size = env.observation_space.shape[0]
        action_size = env.action_space.shape[0]
        model_info = env.model_info()
    finally:
        env.close()
    actor_layers, width, depth = _network(actor, obs_size, action_size, "Actor")
    critic_layers, critic_width, critic_depth = _network(critic, obs_size, 1, "Critic")
    if (width, depth) != (critic_width, critic_depth):
        raise ValueError("Actor and critic must have the same hidden width and depth")
    log_std = _array(log_std, (action_size,), "log_std")
    if (observation_mean is None) != (observation_variance is None):
        raise ValueError("Provide both observation_mean and observation_variance, or omit both")
    mean = _array(np.zeros(obs_size) if observation_mean is None else observation_mean,
                  (obs_size,), "Observation mean")
    variance = _array(np.ones(obs_size) if observation_variance is None else observation_variance,
                      (obs_size,), "Observation variance")
    if np.any(variance < 0):
        raise ValueError("Observation variance must be nonnegative")
    if not np.isfinite(observation_count) or observation_count <= 0:
        raise ValueError("Observation count must be finite and positive")
    if not np.isfinite(observation_clip) or observation_clip <= 0:
        raise ValueError("Observation clip must be finite and positive")
    with np.errstate(over="ignore"):
        m2 = variance * observation_count
    if not np.isfinite(m2).all():
        raise ValueError("Observation variance multiplied by count must remain finite")
    if isinstance(steps, bool) or not isinstance(steps, (int, np.integer)) or steps < 0 or steps > 2**53 - 1:
        raise ValueError("steps must be a nonnegative JavaScript-safe integer")
    config_data = config.model_dump()
    config_data.update(hidden_size=width, hidden_layers=depth, device="auto")
    if name is not None:
        config_data["body"]["name"] = name
    config = TrainConfig.model_validate(config_data)
    weights = {}
    for head, layers in (("actor", actor_layers), ("critic", critic_layers)):
        for index, layer in enumerate(layers):
            stem = f"{head}_out" if index == depth else f"{head}"
            suffix = "" if index == depth else str(index + 1)
            # PyTorch stores [output, input]; TensorFlow.js uses [input, output].
            weights[f"{stem}_w{suffix}"] = _weight(layer.weight.T, f"{head} weight")
            bias = layer.bias if layer.bias is not None else np.zeros(layer.out_features)
            weights[f"{stem}_b{suffix}"] = _weight(bias, f"{head} bias")
    weights["log_std"] = _weight(log_std, "log_std")
    policy_config = {
        key: getattr(config, key)
        for key in ("seed", "hidden_size", "hidden_layers", "epochs", "batch_size",
                    "learning_rate", "clip_range", "entropy_coef")
    }
    policy_config.update(value_coef=0.5, max_grad_norm=0.5)
    record = {
        "format": "morpholab-browser-run-v1",
        "id": f"portable-{uuid.uuid4().hex}",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "stopped", "steps": int(steps), "checkpoint_available": True,
        "config": config.model_dump(),
        "policy": {
            "format": "morpholab-browser-ppo-v1",
            "obsSize": obs_size, "actionSize": action_size,
            "config": policy_config, "randomState": config.seed, "weights": weights,
        },
        "normalizer": {
            "size": obs_size, "clip": float(observation_clip),
            "count": float(observation_count), "mean": mean.tolist(), "m2": m2.tolist(),
        },
        "body_xml": export_mjcf(config.body, config.environment),
        "model_info": model_info,
    }
    text = json.dumps(record, allow_nan=False, separators=(",", ":"))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text + "\n", encoding="utf-8")
    return record


def export_checkpoint(run_dir: Path, output: Path, *, checkpoint: str = "latest") -> dict:
    """Convert a native RLsim SB3 checkpoint without modifying its run folder."""
    from stable_baselines3.common.torch_layers import FlattenExtractor

    config, model, normalizer = (load_checkpoint(Path(run_dir)) if checkpoint == "latest"
                                 else load_checkpoint(Path(run_dir), checkpoint))
    try:
        policy = model.policy
        if getattr(policy, "squash_output", False) or getattr(policy, "use_sde", False):
            raise ValueError("Squashed and state-dependent-noise policies are not supported")
        if any(type(getattr(policy, name, None)) is not FlattenExtractor for name in (
            "features_extractor", "pi_features_extractor", "vf_features_extractor",
        )):
            raise ValueError("Export requires flat observations without a custom feature extractor")
        actor = nn.Sequential(*policy.mlp_extractor.policy_net.children(), policy.action_net)
        critic = nn.Sequential(*policy.mlp_extractor.value_net.children(), policy.value_net)
        # A legacy config lacks architecture fields. Inspect the actual saved
        # model, including its optimizer settings, before exporting metadata.
        settings = config.model_dump()
        settings.update(
            learning_rate=float(model.lr_schedule(1)),
            batch_size=model.batch_size, epochs=model.n_epochs,
            rollout_steps=model.n_steps, gamma=model.gamma,
            entropy_coef=model.ent_coef, clip_range=float(model.clip_range(1)),
        )
        normalization = {"observation_clip": float(np.finfo(np.float32).max)}
        if normalizer.norm_obs:
            if normalizer.epsilon != 1e-8 or isinstance(normalizer.obs_rms, dict):
                raise ValueError("Export requires flat observations and normalization epsilon 1e-8")
            normalization = {
                "observation_mean": normalizer.obs_rms.mean,
                "observation_variance": normalizer.obs_rms.var,
                "observation_count": float(normalizer.obs_rms.count),
                "observation_clip": float(normalizer.clip_obs),
            }
        return export_torch_policy(
            actor, critic, policy.log_std, settings, Path(output),
            steps=model.num_timesteps, **normalization,
        )
    finally:
        normalizer.close()
